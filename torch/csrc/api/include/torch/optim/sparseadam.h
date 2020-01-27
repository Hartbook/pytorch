#pragma once

#include <torch/arg.h>
#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>

#include <utility>
#include <vector>

namespace torch {
namespace serialize {
class OutputArchive;
class InputArchive;
} // namespace serialize
} // namespace torch

namespace torch {
namespace optim {

struct TORCH_API SparseAdamOptions {
  /* implicit */ SparseAdamOptions(double learning_rate);
  TORCH_ARG(double, learning_rate);
  TORCH_ARG(double, beta1) = 0.9;
  TORCH_ARG(double, beta2) = 0.999;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(double, eps) = 1e-8;
  TORCH_ARG(bool, amsgrad) = false;
};

class TORCH_API SparseAdam : public Optimizer {
  public:

  template <typename ParameterContainer>
  explicit SparseAdam(ParameterContainer&& parameters, const SparseAdamOptions& options_)
      : Optimizer(std::forward<ParameterContainer>(parameters)),
        options(options_) {}

  void step() override;

  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

  SparseAdamOptions options;

  std::vector<int64_t> step_buffers;
  std::vector<Tensor> exp_average_buffers;
  std::vector<Tensor> exp_average_sq_buffers;
  std::vector<Tensor> max_exp_average_sq_buffers;

  private:

  SparseAdam() : options(0) {}

  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE(step_buffers);
    _TORCH_OPTIM_SERIALIZE(exp_average_buffers);
    _TORCH_OPTIM_SERIALIZE(exp_average_sq_buffers);
    _TORCH_OPTIM_SERIALIZE(max_exp_average_sq_buffers);
  }
};
} // namespace optim
} // namespace torch

