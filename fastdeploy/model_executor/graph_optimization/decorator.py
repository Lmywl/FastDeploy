"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from abc import abstractmethod
from typing import Callable, Optional, TypeVar

import paddle.nn.layer

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.graph_optimization.graph_optimization_backend import (
    GraphOptBackend,
)

_T = TypeVar("_T", bound=type[paddle.nn.Layer])


def support_graph_optimization(cls: Optional[_T] = None) -> _T:
    """
    A decorator for wrapping models or layers with static graph and CUDAGraph support.
    This enables efficient kernel launch sequencing for improved GPU performance.

    Example usage:

    '''
    @support_graph_optimization
    class ErnieBot(paddle.nn.Layer):
        def __init__(**kwargs):
            ...

        def forward(self, x: paddle.Tensor, y: paddle.Tensor):
            ...
    '''
    """
    if GraphOptWrapper in cls.__bases__:
        return cls
    else:
        # 父类继承, 让传入的 model 继承 GraphOptWrapper
        # 继承链变化为: GraphOptWrapper, paddle.nn.layer -> Ernie
        # MRO(方法解析顺序),调用指定方法时，其查找顺序变为:
        # Ernie -> paddle.nn.layer --> GraphOptWrapper(因为GraphOptWrapper在初始__base__元组的最最后)
        cls.__bases__ = cls.__bases__ + (GraphOptWrapper,)

    origin_init = cls.__init__

    def __init__(self, fd_config: FDConfig, **kwargs):
        """Decorator model.__init__() func"""
        # 新的初始化函数中，除了初始化原始的nn.Layer，还会初始化 GraphOptwrappper ###
        # 为后续调用 图优化后端 做铺垫 ###
        origin_init(self, fd_config=fd_config, **kwargs)
        self.use_graph_opt = fd_config.graph_opt_config.graph_opt_level > 0 or fd_config.graph_opt_config.use_cudagraph
        if self.use_graph_opt:
            GraphOptWrapper.__init__(self, fd_config=fd_config, graph_opt_backend=None)
        else:
            # Not use graph optimization
            return

    def __call__(self, **kwargs):
        """Decorator model.__call__() func"""
        if not self.use_graph_opt:
            return self.forward(**kwargs)
        # 如果开启了 graph_opt, 此时的模型forward控制权会被交到 graph_opt_backend()当中
        return self.graph_opt_backend(**kwargs)

    # 调用该装饰器后，传入cls的__init__方法和__call__方法都会被替
    # 换为自定义的考虑 cudagraph 使用的__init__和__call__方法，
    # 实际上，这里的 __call__ 方法会覆盖掉  GraphOptWrapper.__call__,
    # 这里的 __call__方法修改为 super(GraphOptWrapper, self).__call__(**kwargs) 或许更合适
    cls.__init__ = __init__
    cls.__call__ = __call__
    return cls


class GraphOptWrapper:
    """The wrapper for GraphOptBackend"""

    def __init__(
        self,
        graph_opt_backend: Optional[Callable] = None,
        fd_config: FDConfig = None,
    ):
        if graph_opt_backend is None:
            graph_opt_backend = GraphOptBackend(self.forward, fd_config)
        self.graph_opt_backend = graph_opt_backend

    @abstractmethod
    def forward(self, **kwargs):
        """Abstract methods for implementing model.forward()"""
        pass

    def __call__(self, **kwargs):
        return self.graph_opt_backend(**kwargs)
