// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <limits>

#include <nlohmann/json.hpp>
#include <openvino/runtime/core.hpp>
#include "openvino/genai/generation_config.hpp"
#include "utils.hpp"


namespace ov {
namespace genai {

GenerationConfig::GenerationConfig(const std::string& json_path) {
    using ov::genai::utils::read_json_param;

    std::ifstream f(json_path);
    OPENVINO_ASSERT(f.is_open(), "Failed to open '" + json_path + "' with generation config");

    nlohmann::json data = nlohmann::json::parse(f);
    
    read_json_param(data, "max_new_tokens", max_new_tokens);
    read_json_param(data, "max_length", max_length);
    // note that ignore_eos is not present in HF GenerationConfig
    read_json_param(data, "num_beam_groups", num_beam_groups);
    read_json_param(data, "num_beams", num_beams);
    read_json_param(data, "diversity_penalty", diversity_penalty);
    read_json_param(data, "length_penalty", length_penalty);
    read_json_param(data, "num_return_sequences", num_return_sequences);
    read_json_param(data, "no_repeat_ngram_size", no_repeat_ngram_size);
    read_json_param(data, "temperature", temperature);
    read_json_param(data, "top_p", top_p);
    read_json_param(data, "top_k", top_k);
    read_json_param(data, "do_sample", do_sample);
    read_json_param(data, "repetition_penalty", repetition_penalty);
    read_json_param(data, "eos_token_id", eos_token_id);

    if (data.contains("early_stopping")) {
        auto field_type = data["early_stopping"].type();
        if (field_type == nlohmann::json::value_t::string && data["early_stopping"] == "never") {
            stop_criteria = StopCriteria::NEVER;
        } else if (field_type == nlohmann::json::value_t::boolean && data["early_stopping"] == true) {
            stop_criteria = StopCriteria::EARLY;
        } else if (field_type == nlohmann::json::value_t::boolean && data["early_stopping"] == false) {
            stop_criteria = StopCriteria::HEURISTIC;
        }
    }
}

void GenerationConfig::update_generation_config(const ov::AnyMap& config_map) {
    using ov::genai::utils::read_anymap_param;
    
    read_anymap_param(config_map, "max_new_tokens", max_new_tokens);
    read_anymap_param(config_map, "max_length", max_length);
    read_anymap_param(config_map, "ignore_eos", ignore_eos);
    read_anymap_param(config_map, "num_beam_groups", num_beam_groups);
    read_anymap_param(config_map, "num_beams", num_beams);
    read_anymap_param(config_map, "diversity_penalty", diversity_penalty);
    read_anymap_param(config_map, "length_penalty", length_penalty);
    read_anymap_param(config_map, "num_return_sequences", num_return_sequences);
    read_anymap_param(config_map, "no_repeat_ngram_size", no_repeat_ngram_size);
    read_anymap_param(config_map, "stop_criteria", stop_criteria);
    read_anymap_param(config_map, "temperature", temperature);
    read_anymap_param(config_map, "top_p", top_p);
    read_anymap_param(config_map, "top_k", top_k);
    read_anymap_param(config_map, "do_sample", do_sample);
    read_anymap_param(config_map, "repetition_penalty", repetition_penalty);
    read_anymap_param(config_map, "eos_token_id", eos_token_id);
}

size_t GenerationConfig::get_max_new_tokens(size_t prompt_length) const {
    // max_new_tokens has priority over max_length, only if max_new_tokens was not specified use max_length
    if (max_new_tokens != SIZE_MAX) {
        return max_new_tokens;
    } else {
        return max_length - prompt_length;
    }
}

bool GenerationConfig::is_greedy_decoding() const {
    return !do_sample && !is_beam_search();
}

bool GenerationConfig::is_beam_search() const {
    return num_beams > 1;
}

bool GenerationConfig::is_multinomial() const {
    return do_sample;
}

void GenerationConfig::validate() const {
    OPENVINO_ASSERT(!do_sample || num_beams == 1, 
                    "Beam search with sampling is not supported yet. "
                    "Please either set do_sample=false to use beam search "
                    "or set num_beams=1 if you with to use multinomial sampling.");
    
    OPENVINO_ASSERT(max_new_tokens > 0, "'max_new_tokens' must be greater than 0");
    
    // max_new_tokens has priority over max_length
    // if max_new_tokens is defined no need to check max_length
    OPENVINO_ASSERT(max_new_tokens != SIZE_MAX ||  max_length > 0, 
                    "'max_length' must be greater than 0 or 'max_new_tokens' should be defined");

    OPENVINO_ASSERT(!do_sample || top_k > 0,
                    "top_k must be a strictly positive, but got ",
                    top_k);
    OPENVINO_ASSERT(!do_sample || (top_p > 0 && top_p <= 1.0f),
                    "top_p must be a positive float > 0 and < 1, but got ",
                    top_p);
    OPENVINO_ASSERT(!do_sample || temperature > 0,
                    "Temperature must be a strictly positive float, but got ",
                    temperature);

    OPENVINO_ASSERT(repetition_penalty > 0,
                    "Repetition penalty must be a strictly positive float, but got ",
                    repetition_penalty);
    
    OPENVINO_ASSERT(!ignore_eos || max_new_tokens != SIZE_MAX || max_length != SIZE_MAX,
                    "ignore_eos == true, in this case either 'max_new_tokens', or 'max_length' should be defined.");

    OPENVINO_ASSERT(eos_token_id != -1 || max_new_tokens != SIZE_MAX || max_length != SIZE_MAX,
                    "Either 'eos_token_id', or 'max_new_tokens', or 'max_length' should be defined.");
}

}  // namespace genai
}  // namespace ov
