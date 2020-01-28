#include <torch/optim/sparseadam.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/nn/module.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <ATen/ATen.h>

#include <cmath>
#include <functional>

namespace torch {
namespace optim {
SparseAdamOptions::SparseAdamOptions(double learning_rate)
    : learning_rate_(learning_rate) {}

void SparseAdam::step() {
  for (size_t i = 0; i < parameters_.size(); ++i)
  {
    Tensor p = parameters_.at(i);
    if (!p.grad().defined()) {
      continue;
    }

    if (!p.grad().is_sparse()) {
      TORCH_CHECK(p.grad().is_sparse(), "SparseAdam does not support dense gradients, please consider Adam instead");
    }

    auto& exp_average = buffer_at(exp_average_buffers, i);
    auto& exp_average_sq = buffer_at(exp_average_sq_buffers, i);

    buffer_at(step_buffers, i) += 1;
    const auto bias_correction1 = 1 - std::pow(options.beta1(), buffer_at(step_buffers, i));
    const auto bias_correction2 = 1 - std::pow(options.beta2(), buffer_at(step_buffers, i));

    NoGradGuard guard;
    p.grad() = p.grad().coalesce();
    auto indices = p.grad().indices().squeeze();
    auto values = p.grad().values();

    auto old_exp_average_values = exp_average.sparse_mask(p.grad())._values();
    auto exp_average_update_values = values.sub(old_exp_average_values).mul_(1 - options.beta1());
    for (unsigned int j = 0; j < indices.size(0); j++) {
      exp_average[indices[j].item<long>()] += exp_average_update_values[j];
    }
    auto old_exp_average_sq_values = exp_average_sq.sparse_mask(p.grad())._values();
    auto exp_average_sq_update_values = values.pow(2).sub_(old_exp_average_sq_values).mul_(1 - options.beta2());
    for (unsigned int j = 0; j < indices.size(0); j++) {
      exp_average_sq[indices[j].item<long>()] += exp_average_sq_update_values[j];
    }

    auto numer = exp_average_update_values.add_(old_exp_average_values);
    exp_average_sq_update_values.add_(old_exp_average_sq_values);
    auto denom = exp_average_sq_update_values.sqrt_().add_(options.eps());
    const auto step_size = options.learning_rate() * std::sqrt(bias_correction2) / bias_correction1;
    auto divided = numer.div(denom);
    for (unsigned int j = 0; j < indices.size(0); j++) {
      p.data()[indices[j].item<long>()] += -step_size*divided[j];
    }
  }
}

void SparseAdam::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void SparseAdam::load(serialize::InputArchive& archive) {
  serialize(*this, archive);
}
} // namespace optim
} // namespace torch

