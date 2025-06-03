from abc import ABC


class RLHFTrainHook(ABC):
    def on_train_start(self) -> None:
        """
        整体模型训练开始
        """
        pass

    def on_train_end(self, global_step: int) -> None:
        """
        整体模型训练结束
        """
        pass

    def on_experience_learn_start(self, experience_step: int) -> None:
        """
        一次experience学习的开始
        一次experience = 一次experience make +  一次experience train
        """
        pass

    def on_experience_learn_end(self, experience_step: int) -> None:
        """
           一次experience学习的开始结束
        """
        pass

    def on_experience_make_start(self, experience_step: int) -> None:
        """
           一次experience产生开始
        """
        pass

    def on_experience_make_end(self, experience_step: int) -> None:
        """
           一次experience产生结束
        """
        pass

    def on_experience_train_start(self, experience_step: int) -> None:
        """
           一次experience的训练开始
        """
        pass

    def on_experience_train_end(self, experience_step: int, global_step: int, metrics: dict) -> None:
        """
           一次experience的训练结束
        """
        pass

    def on_experience_train_batch_start(self, experience_step: int, global_step: int) -> None:
        """
            等同于step_start
        """
        pass

    def on_experience_train_batch_end(self, experience_step: int, global_step: int, metrics: dict) -> None:
        """
            等同于step_end
        """
        pass

    def on_experience_batch_start(self, experience_step: int) -> None:
        pass

    def on_experience_batch_end(self, experience_step: int) -> None:
        pass


class SFTTrainHook(ABC):
    """多继承时，小心python mro问题"""
    def on_train_start(self) -> None:
        """
        整体模型训练开始
        """
        pass

    def on_train_end(self, global_step: int) -> None:
        """
        整体模型训练结束
        """
        pass

    def on_step_train_start(self, global_step: int) -> None:
        """
           一次训练开始
        """
        pass

    def on_step_train_end(self, global_step: int, metrics: dict = None) -> None:
        """
           一次训练结束
        """
        pass
