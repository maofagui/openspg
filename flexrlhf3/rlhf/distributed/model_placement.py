from alignment.api.rlhf.config import Placement, InitialRewardSeparate, AllCollocate, Slot
from alignment.api.utils import dist_util


def _find_one_sum_1_pairs(model_ratio_list):
    result = []
    for i in range(len(model_ratio_list)):
        if model_ratio_list[i] == 100:
            sub_result = []
            sub_result.append(i)
            result.append(sub_result)
    return result


def _find_two_sum_1_pairs(model_ratio_list):
    result = []
    length = len(model_ratio_list)
    for i in range(length):
        for j in range(i + 1, length):
            if model_ratio_list[i] + model_ratio_list[j] == 100:
                current_pair = [i, j]
                result.append(current_pair)
    return result


def _find_three_sum_1_pairs(model_ratio_list):
    length = len(model_ratio_list)
    for i in range(length):
        for j in range(i + 1, length):
            for k in range(j + 1, length):
                if model_ratio_list[i] + model_ratio_list[j] + model_ratio_list[k] == 100:
                    result = [i, j, k]
                    return result

    return []


def _find_four_sum_1_pairs(model_ratio_list):
    length = len(model_ratio_list)
    for i in range(length):
        for j in range(i + 1, length):
            for k in range(j + 1, length):
                for l in range(k + 1, length):
                    sum = model_ratio_list[i] + model_ratio_list[j] + model_ratio_list[k] + model_ratio_list[l]
                    if sum == 100:
                        result = [i, j, k, l]
                        return result

    return []


def _find_all_sum_1(model_ratio_list):
    """
        model_ratio_list : [1, 1, 0.5, 0.5]

        return : [0], [1], [2, 3]
    """
    assert len(model_ratio_list) <= 4, "only 4 model can be place"

    model_ratio_list_scale = []
    for i in range(len(model_ratio_list)):
        model_ratio_list_scale.append(100 * model_ratio_list[i])

    one_sum_1 = _find_one_sum_1_pairs(model_ratio_list_scale)
    two_sum_1 = _find_two_sum_1_pairs(model_ratio_list_scale)
    three_sum_1 = _find_three_sum_1_pairs(model_ratio_list_scale)
    four_sum_1 = _find_four_sum_1_pairs(model_ratio_list_scale)

    result = []
    if len(one_sum_1) > 0:
        result.extend(one_sum_1)
    if len(two_sum_1) > 0:
        for two_sum_ele in two_sum_1:
            if len(two_sum_ele) > 0:
                result.append(two_sum_ele)
    if len(three_sum_1) > 0:
        result.append(three_sum_1)
    if len(four_sum_1) > 0:
        result.append(four_sum_1)

    return result


def assign_rank(models, model_ratio_list, all_world_size):
    """
    [2, 3], [0.5, 0.5]
    """
    result = []
    for i in range(len(models)):
        model_index = models[i]
        model_ratio = model_ratio_list[model_index]

        rank_size = all_world_size * model_ratio

        ranks = []
        for j in range(int(rank_size)):
            ranks.append(int(rank_size * i + j))
        result.append(ranks)
    return result


class ModelPlacement:
    def __init__(self,
                 placement: Placement,
                 actor,
                 critic,
                 init_model,
                 reward_model,
                 all_world_size=dist_util.get_total_world_size()):
        self.placement = placement
        self.actor = actor
        self.critic = critic
        self.init_model = init_model
        self.reward_model = reward_model

        self.models_ranks = getattr(placement, 'model_ranks', None)
        if self.models_ranks is None:
            self._place_models_to_ranks(all_world_size)

    def get_actor_ranks(self):
        if isinstance(self.models_ranks, list):
            return self.models_ranks[0]
        return self.models_ranks.actor_ranks

    def get_critic_ranks(self):
        if isinstance(self.models_ranks, list):
            return self.models_ranks[1]
        return self.models_ranks.critic_ranks

    def get_init_model_ranks(self):
        if isinstance(self.models_ranks, list):
            return self.models_ranks[2]
        return self.models_ranks.init_model_ranks

    def get_reward_model_ranks(self):
        if isinstance(self.models_ranks, list):
            return self.models_ranks[3]
        return self.models_ranks.reward_model_ranks

    def get_pred_actor_ranks(self):
        if isinstance(self.models_ranks, list):
            # 暂未包含策略，=actor_ranks            
            return self.models_ranks[0]
        return self.models_ranks.pred_actor_ranks

    def get_pred_critic_ranks(self):
        if isinstance(self.models_ranks, list):
            # 暂未包含策略，=actor_ranks            
            return self.models_ranks[1]
        return self.models_ranks.pred_critic_ranks

    def _place_models_to_ranks(self, all_world_size):
        if all_world_size == 1:
            self.models_ranks = [[0], [0], [0], [0]]
            return
        self.models_ranks = [None, None, None, None]
        slot = self.placement.slot

        actor_ratio = slot.actor_ratio
        if self.critic is None:
            critic_ratio = 0
        else:
            critic_ratio = slot.critic_ratio
        if self.init_model is None:
            initial_ratio = 0
        else:
            initial_ratio = slot.initial_ratio

        if self.reward_model is None:
            reward_ratio = 0
        else:
            reward_ratio = slot.reward_ratio

        assert (actor_ratio + critic_ratio + reward_ratio +
                initial_ratio) % 1 == 0, "all ratio in slot sum must multiple 1"

        model_ratio_list = [actor_ratio, critic_ratio, initial_ratio, reward_ratio]

        model_group = _find_all_sum_1(model_ratio_list)

        # [0, 1, [2, 3]]
        for i in range(len(model_group)):
            model_assign_ranks = assign_rank(model_group[i], model_ratio_list, all_world_size)
            for j in range(len(model_group[i])):
                model_index = model_group[i][j]
                model_index_ranks = model_assign_ranks[j]

                self.models_ranks[model_index] = model_index_ranks


if __name__ == '__main__':
    """ 
    model_ratio_list = [1, 1, 1, 1]
    print(_find_all_sum_1(model_ratio_list))
    print("========================")

    model_ratio_list = [1, 0.5, 0.5, 1]
    print(_find_all_sum_1(model_ratio_list))
    print("========================")

    model_ratio_list = [1, 0.5, 0.3, 0.2]
    print(_find_all_sum_1(model_ratio_list))
    print("========================")

    model_ratio_list = [0.1, 0.5, 0.3, 0.1]
    print(_find_all_sum_1(model_ratio_list))
    print("========================")

    model_ratio_list = [0.5, 0.5, 0.8, 0.2]
    print(_find_all_sum_1(model_ratio_list))
    print("========================")
        
    """

    placement = InitialRewardSeparate()
    model = ModelPlacement(placement, 'ss', 'ss', 'ss', 'ss', 8)
    print(model.models_ranks)

    placement = AllCollocate()
    model = ModelPlacement(placement, 'ss', 'ss', 'ss', 'ss', 8)
    print(model.models_ranks)

    slot = Slot(0.25, 0.25, 0.25, 0.25)
    placement = Placement(slot)

    model = ModelPlacement(placement, 'ss', 'ss', 'ss', 'ss', 8)
    print(model.models_ranks)
