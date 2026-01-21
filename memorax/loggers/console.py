from collections import defaultdict
from dataclasses import field
from typing import DefaultDict

from flax import struct

from .logger import BaseLogger, BaseLoggerState, PyTree


@struct.dataclass(frozen=True)
class ConsoleLoggerState(BaseLoggerState):
    buffer: DefaultDict[int, dict[str, PyTree]] = field(
        default_factory=lambda: defaultdict(dict)
    )


@struct.dataclass(frozen=True)
class ConsoleLogger(BaseLogger[ConsoleLoggerState]):
    def init(self, cfg) -> ConsoleLoggerState:
        return ConsoleLoggerState()

    def log(
        self, state: ConsoleLoggerState, data: PyTree, step: int
    ) -> ConsoleLoggerState:
        state.buffer[step].update(data)
        return state

    def _strong_line(self):
        print("###############################################")

    def _weak_line(self):
        print("-----------------------------------------------")

    def emit(self, state: ConsoleLoggerState) -> ConsoleLoggerState:
        for step, data in sorted(state.buffer.items()):
            training = {
                k.split("/")[-1]: v.mean().item()
                for k, v in data.items()
                if k.startswith("training")
            }
            evaluation = {
                k.split("/")[-1]: v.mean().item()
                for k, v in data.items()
                if k.startswith("evaluation")
            }
            losses = {
                k.split("/")[-1]: v.mean().item()
                for k, v in data.items()
                if k.startswith("loss")
            }

            if training:
                self._strong_line()
                print(f"TRAINING - {step:_}")
                self._strong_line()

                longest_key = len(max(training, key=len)) + 5
                for k, v in training.items():
                    k += ":"
                    print(f"{k:<{longest_key}}: {v:.2f}")
                    self._weak_line()
            if evaluation:
                self._strong_line()
                print(f"EVALUATION - {step:_}")
                self._strong_line()
                longest_key = len(max(evaluation, key=len)) + 5
                for k, v in evaluation.items():
                    k += ":"
                    print(f"{k:<{longest_key}} {v:.2f}")
                    self._weak_line()
            if losses:
                self._strong_line()
                print(f"LOSSES - {step:_}")
                self._strong_line()
                longest_key = len(max(losses, key=len)) + 5
                for k, v in losses.items():
                    k += ":"
                    print(f"{k:<{longest_key}}: {v:.2f}")
                    self._weak_line()

        state.buffer.clear()
        return state
