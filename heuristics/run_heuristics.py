from heuristics.heuristics_interval_owf import HeuristicsOwf
from heuristics.heuristics_intervals_struct import HeuristicsStruct
from heuristics.heuristics_intervals_zayas import HeuristicsZayas
import timeit
from functools import partial

if __name__ == '__main__':

    search = False
    env = "zayas"  # "struct" or "owf" or "zayas"
    eval_size = 1500

    if env == "struct":
        n_comp = 5
        discount_reward = 0.95
        k_comp = 4
        env_correlation = False
        campaign_cost = False
        seed_test = 0
        heuristic = HeuristicsStruct(n_comp=n_comp,
                                     discount_reward=discount_reward,
                                     k_comp=k_comp,
                                     env_correlation=env_correlation,
                                     campaign_cost=campaign_cost,
                                     seed=seed_test)
        eval_part = partial(heuristic.eval)
    elif env == "owf":
        n_owt = 2
        discount_reward = 0.95
        lev = 3
        campaign_cost = False
        seed_test = 0
        heuristic = HeuristicsOwf(n_owt=n_owt,
                                  lev=lev,
                                  discount_reward=discount_reward,
                                  campaign_cost=False,
                                  seed=seed_test)
        eval_part = partial(heuristic.eval)
    elif env == "zayas":
        n_comp = 22
        freq_col = [1.e-2, 1.e-3, 5.e-4, 1.e-5, 1.e-6]
        discount_reward = 0.95
        seed_test = 0
        eval_size = 2000

        #### Evaluation
        pf_brace_rep = 0.01

        heuristic = HeuristicsZayas(n_comp=n_comp,
                                    # Number of structure
                                    freq_col=freq_col,
                                    discount_reward=discount_reward,
                                    campaign_cost=False,
                                    seed=seed_test)
        eval_part = partial(heuristic.eval, pf_sys_rep=pf_brace_rep)
    else:
        heuristic = None

    if search:
        #### Search
        starting_time = timeit.default_timer()
        heuristic.search(eval_size)
        print("Time (s):", timeit.default_timer() - starting_time)

    else:
        #### Evaluation
        insp_int = 10
        insp_comp = 5
        eval_part(eval_size, insp_int, insp_comp)
