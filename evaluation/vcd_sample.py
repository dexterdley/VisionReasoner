import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import math
from torch import nn
import torch.nn.functional as F
import pdb
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers

try:
    from transformers.generation.utils import (
        GenerateDecoderOnlyOutput,
        GenerateEncoderDecoderOutput,
        SampleOutput, SampleEncoderDecoderOutput, SampleDecoderOnlyOutput,
    )
except ImportError:
    GenerateDecoderOnlyOutput = Dict[str, Any]
    GenerateEncoderDecoderOutput = Dict[str, Any]
    SampleOutput = Dict[str, Any]
    SampleEncoderDecoderOutput = Dict[str, Any]
    SampleDecoderOnlyOutput = Dict[str, Any]

logits_processor = LogitsProcessorList()
logits_warper = LogitsProcessorList()
IMAGE_TOKEN_INDEX = -200
import os
from transformers import GenerationConfig
from transformers.generation.utils import GenerationMixin

def add_diffusion_noise(image_tensor, noise_step):
    """Add diffusion-schedule noise to a tensor at a given timestep (0-999)."""
    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    noise = torch.randn_like(image_tensor)
    alphas_t = alphas_bar_sqrt[noise_step]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[noise_step]
    return alphas_t * image_tensor + alphas_1_m_t * noise

class VisualZeroHook:
    def __init__(self, start_idx, end_idx):
        self.start = start_idx
        self.end = end_idx

    def __call__(self, module, args):
        hidden_states = args[0]
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        B = hidden_states.shape[0]
        half = B // 2
        hidden_states[half:, self.start:self.end, :].zero_()
        return (hidden_states,) + args[1:]

class VisualTextDistortionHook:
    """
    Hook to add Gaussian noise to embeddings.
    "vis" for VCD (Visual Contrastive Decoding)
    "text" for ICD (Instruction Contrastive Decoding)
    """
    def __init__(self, start_idx, end_idx, noise_type="vis", noise_scale=0.1):
        self.start = start_idx
        self.end = end_idx
        self.noise_type = noise_type
        self.noise_scale = noise_scale

    def __call__(self, module, args):
        hidden_states = args[0]
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        B = hidden_states.shape[0]
        half = B // 2
        if self.noise_type == "vis":
            noise = torch.randn_like(hidden_states[half:, self.start:self.end, :]) * self.noise_scale
            hidden_states[half:, self.start:self.end, :] += noise

        elif self.noise_type == "text":
            noise_pre = torch.randn_like(hidden_states[half:, :self.start, :]) * self.noise_scale
            hidden_states[half:, :self.start, :] += noise_pre
                
            noise_post = torch.randn_like(hidden_states[half:, self.end:, :]) * self.noise_scale
            hidden_states[half:, self.end:, :] += noise_post
        return (hidden_states,) + args[1:]

def _sample_vord(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:

        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            if self.config._attn_implementation == "flash_attention_2":
                if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        visual_alpha = model_kwargs.get('visual_alpha', getattr(generation_config, 'visual_alpha', 0.0))

        # ------------------------------------------------------------------
        # Setup Double Pass Kwargs
        # ------------------------------------------------------------------
        model_kwargs_clean = model_kwargs.copy()
        model_kwargs_noisy = model_kwargs.copy()

        # Prevent KV Cache collisions between passes if cache already exists
        import copy
        if "past_key_values" in model_kwargs and model_kwargs["past_key_values"] is not None:
            model_kwargs_noisy["past_key_values"] = copy.deepcopy(model_kwargs["past_key_values"])

        noise_step = int(getattr(self, 'noise_step', model_kwargs.get('noise_step', 500)))
        margin = torch.tensor(0.0, device=input_ids.device)  # default if no pixel values
        for pixel_key in ["pixel_values", "pixel_values_videos"]:
            if pixel_key in model_kwargs_noisy and model_kwargs_noisy[pixel_key] is not None:
                p_clean = model_kwargs_clean[pixel_key]
                p_noisy = add_diffusion_noise(p_clean, noise_step)
                model_kwargs_noisy[pixel_key] = p_noisy

                # Adaptive margin via ViT embeddings: m = (1/π) * arccos(cos_sim(f(v), f(v̂)))
                grid_thw = model_kwargs_clean.get("image_grid_thw", model_kwargs_clean.get("video_grid_thw", None))
                with torch.no_grad():
                    f_clean = self.visual(p_clean, grid_thw=grid_thw)   # [N_tokens, D]
                    f_noisy = self.visual(p_noisy, grid_thw=grid_thw)   # [N_tokens, D]
                # Mean-pool over tokens to get a single feature vector per image
                f_clean = f_clean.mean(dim=0 if f_clean.dim() == 2 else 1).flatten()
                f_noisy = f_noisy.mean(dim=0 if f_noisy.dim() == 2 else 1).flatten()
                cos_sim = F.cosine_similarity(
                    f_clean.unsqueeze(0).float(),
                    f_noisy.unsqueeze(0).float(),
                    dim=-1
                ).clamp(-1, 1)
                margin = (1.0 / math.pi) * torch.acos(cos_sim)  # scalar in [0, 1]
        # ------------------------------------------------------------------

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            forward_call = self if is_prefill else model_forward
            
            # Pass 1: Clean
            model_inputs_clean = self.prepare_inputs_for_generation(input_ids, **model_kwargs_clean)
            outputs_clean = forward_call(**model_inputs_clean, return_dict=True)
            
            # Pass 2: Noisy
            model_inputs_noisy = self.prepare_inputs_for_generation(input_ids, **model_kwargs_noisy)
            outputs_noisy = forward_call(**model_inputs_noisy, return_dict=True)

            # Extract logits 
            next_token_logits = outputs_clean.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_logits_cd = outputs_noisy.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # VORD Logic:
            # Calculate Adaptive Plausibility Constraints and ordinal mask
            cutoff = torch.log(torch.tensor(0.1)) + next_token_logits.max(dim=-1, keepdim=True).values
            ordinal_mask = (F.softmax(next_token_logits,-1) + margin).clamp(max=1) >= F.softmax(next_token_logits_cd,-1)

            # Use the correctly contrasted logits
            next_token_scores = logits_processor(input_ids, next_token_logits)

            if is_prefill:
                is_prefill = False

            if synced_gpus and this_peer_finished:
                continue

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs_clean.decoder_attentions,) if self.config.is_encoder_decoder else (outputs_clean.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs_clean.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs_clean.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs_clean.hidden_states,)
                    )

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                #probs[next_token_logits < cutoff] = 0
                probs[~ordinal_mask] = 0
                # Safety fallback: if mask zeroed everything, fall back to clean probs
                prob_sums = probs.sum(dim=-1, keepdim=True)
                all_zero = (prob_sums == 0)
                if all_zero.any():
                    fallback_probs = nn.functional.softmax(next_token_scores, dim=-1)
                    probs = torch.where(all_zero, fallback_probs, probs)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            if input_ids.shape[0] != next_tokens.shape[0] and next_tokens.shape[0] == 1:
                next_tokens = next_tokens.repeat(input_ids.shape[0])

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # Update KV Cache independently for BOTH passes
            model_kwargs_clean = self._update_model_kwargs_for_generation(
                outputs_clean,
                model_kwargs_clean,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            model_kwargs_noisy = self._update_model_kwargs_for_generation(
                outputs_noisy,
                model_kwargs_noisy,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1
            
            del outputs_clean
            del outputs_noisy

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs_clean.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs_clean.get("past_key_values"),
                )
        else:
            return input_ids

def _sample_vgd(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:

        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            if self.config._attn_implementation == "flash_attention_2":
                if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        visual_alpha = model_kwargs.get('visual_alpha', getattr(generation_config, 'visual_alpha', 0.0))

        # ------------------------------------------------------------------
        # Setup Double Pass Kwargs
        # ------------------------------------------------------------------
        model_kwargs_clean = model_kwargs.copy()
        model_kwargs_noisy = model_kwargs.copy()

        # Prevent KV Cache collisions between passes if cache already exists
        import copy
        if "past_key_values" in model_kwargs and model_kwargs["past_key_values"] is not None:
            model_kwargs_noisy["past_key_values"] = copy.deepcopy(model_kwargs["past_key_values"])

        noise_scale = getattr(generation_config, 'noise_scale', 0.1)
        for pixel_key in ["pixel_values", "pixel_values_videos"]:
            if pixel_key in model_kwargs_noisy and model_kwargs_noisy[pixel_key] is not None:
                p_vals = model_kwargs_noisy[pixel_key].clone()
                
                # Assign a completely NEW tensor so we don't zero the clean branch
                # Choose either zeros or noise based on your preference
                model_kwargs_noisy[pixel_key] = torch.zeros_like(p_vals) 
        # ------------------------------------------------------------------

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            forward_call = self if is_prefill else model_forward
            
            # Pass 1: Clean
            model_inputs_clean = self.prepare_inputs_for_generation(input_ids, **model_kwargs_clean)
            outputs_clean = forward_call(**model_inputs_clean, return_dict=True)
            
            # Pass 2: Noisy
            model_inputs_noisy = self.prepare_inputs_for_generation(input_ids, **model_kwargs_noisy)
            outputs_noisy = forward_call(**model_inputs_noisy, return_dict=True)

            # Extract logits 
            next_token_logits_clean = outputs_clean.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_logits_noisy = outputs_noisy.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # VGD Logic Corrected: Clean + alpha * (Clean - Noisy)
            vgd_logits = next_token_logits_clean + visual_alpha * (next_token_logits_clean - next_token_logits_noisy)
            
            # Use the correctly contrasted logits
            next_token_scores = logits_processor(input_ids, vgd_logits)

            if is_prefill:
                is_prefill = False

            if synced_gpus and this_peer_finished:
                continue

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits_clean,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs_clean.decoder_attentions,) if self.config.is_encoder_decoder else (outputs_clean.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs_clean.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs_clean.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs_clean.hidden_states,)
                    )

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            if input_ids.shape[0] != next_tokens.shape[0] and next_tokens.shape[0] == 1:
                next_tokens = next_tokens.repeat(input_ids.shape[0])

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # Update KV Cache independently for BOTH passes
            model_kwargs_clean = self._update_model_kwargs_for_generation(
                outputs_clean,
                model_kwargs_clean,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            model_kwargs_noisy = self._update_model_kwargs_for_generation(
                outputs_noisy,
                model_kwargs_noisy,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1
            
            del outputs_clean
            del outputs_noisy

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs_clean.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs_clean.get("past_key_values"),
                )
        else:
            return input_ids

def _sample_contrastive(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:

        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            if self.config._attn_implementation == "flash_attention_2":
                if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        visual_alpha = model_kwargs.get('visual_alpha', getattr(generation_config, 'visual_alpha', 0.0))

        # ------------------------------------------------------------------
        # Setup Double Pass Kwargs
        # ------------------------------------------------------------------
        model_kwargs_clean = model_kwargs.copy()
        model_kwargs_noisy = model_kwargs.copy()

        # Prevent KV Cache collisions between passes if cache already exists
        import copy
        if "past_key_values" in model_kwargs and model_kwargs["past_key_values"] is not None:
            model_kwargs_noisy["past_key_values"] = copy.deepcopy(model_kwargs["past_key_values"])

        noise_step = int(getattr(self, 'noise_step', model_kwargs.get('noise_step', 500)))
        margin = torch.tensor(0.0, device=input_ids.device)  # default if no pixel values
        for pixel_key in ["pixel_values", "pixel_values_videos"]:
            if pixel_key in model_kwargs_noisy and model_kwargs_noisy[pixel_key] is not None:
                p_clean = model_kwargs_clean[pixel_key]
                p_noisy = add_diffusion_noise(p_clean, noise_step)
                model_kwargs_noisy[pixel_key] = p_noisy
        # ------------------------------------------------------------------

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            forward_call = self if is_prefill else model_forward
            
            # Pass 1: Clean
            model_inputs_clean = self.prepare_inputs_for_generation(input_ids, **model_kwargs_clean)
            outputs_clean = forward_call(**model_inputs_clean, return_dict=True)
            
            # Pass 2: Noisy
            model_inputs_noisy = self.prepare_inputs_for_generation(input_ids, **model_kwargs_noisy)
            outputs_noisy = forward_call(**model_inputs_noisy, return_dict=True)

            # Extract logits 
            next_token_logits = outputs_clean.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_logits_cd = outputs_noisy.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # VCD Logic:
            # Calculate Adaptive Plausibility Constraints and ordinal mask
            cutoff = torch.log(torch.tensor(0.1)) + next_token_logits.max(dim=-1, keepdim=True).values

            vcd_logits = 2 * next_token_logits - next_token_logits_cd
            # Use the correctly contrasted logits
            next_token_scores = logits_processor(input_ids, vcd_logits)

            if is_prefill:
                is_prefill = False

            if synced_gpus and this_peer_finished:
                continue

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs_clean.decoder_attentions,) if self.config.is_encoder_decoder else (outputs_clean.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs_clean.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs_clean.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs_clean.hidden_states,)
                    )

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            if input_ids.shape[0] != next_tokens.shape[0] and next_tokens.shape[0] == 1:
                next_tokens = next_tokens.repeat(input_ids.shape[0])

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # Update KV Cache independently for BOTH passes
            model_kwargs_clean = self._update_model_kwargs_for_generation(
                outputs_clean,
                model_kwargs_clean,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            model_kwargs_noisy = self._update_model_kwargs_for_generation(
                outputs_noisy,
                model_kwargs_noisy,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1
            
            del outputs_clean
            del outputs_noisy

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs_clean.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs_clean.get("past_key_values"),
                )
        else:
            return input_ids

def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            if self.config._attn_implementation == "flash_attention_2":
                if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

_original_validate_model_kwargs = GenerationMixin._validate_model_kwargs

def _validate_model_kwargs_vgd(self, model_kwargs: Dict[str, Any]):
    ignore_keys = {
        'visual_alpha', 'mode', 'vcd_alpha', 'icd_alpha', 'verbose', 'reuse', 'use_custom_prompt',
        'use_vllm', 'presence_penalty', 'repetition_penalty', 'top_p', 'top_k',
        'vgd_input_ids'
    }
    removed = {}
    for k in list(model_kwargs.keys()):
        if k in ignore_keys:
            removed[k] = model_kwargs.pop(k)
    try:
        output = _original_validate_model_kwargs(self, model_kwargs)
    finally:
        model_kwargs.update(removed)
    if output is not None and isinstance(output, dict) and output is not model_kwargs:
        output.update(removed)
        return output
    return model_kwargs

def evolve_guidance_sampling(visual_alpha=0.0, mode="VORD"):
    """
    Monkey-patch HF's _sample method to use the appropriate decoding strategy.
    
    In transformers 4.50+, generate() always routes through _sample() regardless
    of do_sample. The do_sample flag is handled inside _sample (argmax vs multinomial).
    """
    transformers.generation.utils.GenerationMixin._validate_model_kwargs = _validate_model_kwargs_vgd
    
    if mode == "VORD":
        print(f"Using VORD Sampling | Alpha: {visual_alpha}")
        transformers.generation.utils.GenerationMixin._sample = _sample_vord
    elif mode == "VGD":
        print(f"Using VGD Sampling | Alpha: {visual_alpha}")
        transformers.generation.utils.GenerationMixin._sample = _sample_vgd
    elif mode == "VCD":
        print(f"Using VCD Sampling  | Alpha: {visual_alpha}")
        transformers.generation.utils.GenerationMixin._sample = _sample_contrastive
    elif mode == "ICD":
        print(f"Using ICD Sampling  | Alpha: {visual_alpha}")
        transformers.generation.utils.GenerationMixin._sample = _sample_contrastive
    else:
        print("Using Regular Sampling")
        transformers.generation.utils.GenerationMixin._sample = _sample
