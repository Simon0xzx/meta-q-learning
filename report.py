import matplotlib.pyplot as plt
import csv
import os.path as path
def read_csv(dir):
    timestamp = []
    ep_reward_mean = []
    with open(dir, newline='\n') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            timestamp.append(int(row['total_timesteps']))
            ep_reward_mean.append(float(row['eval_eprewmean']))
    return timestamp, ep_reward_mean

def plot_curve(axs, dir, label):
    timestamp, ep_reward_mean = read_csv(dir)
    axs.plot(timestamp, ep_reward_mean, lw=2, label=label)

def report():
    row, col = 1, 3
    fig, axs = plt.subplots(row, col)
    # Push
    axs[0].set_title('MQL ML1 Push')
    axs[0].set_xlabel('Env Steps')
    axs[0].set_ylabel('Average Test Reward')

    mql_push_dir = '/home/simon0xzx/research/berkely_research/meta-q-learning/result/mql_ml1/metaworld-ml1-push-v1_mql_default/eval.csv'
    plot_curve(axs[0], mql_push_dir, "MQL-push")

    # tclmql_push_dir = "/home/simon0xzx/research/berkely_research/meta-q-learning/result/tcl_mql_ml1/metaworld-ml1-push-v1_mql_tcl_first/eval.csv"
    # plot_curve(axs[0], tclmql_push_dir, "TCL-MQL-push-struct")
    mql_struct_dir = '/home/simon0xzx/research/berkely_research/meta-q-learning/result/tcl_mql_ml1/metaworld-ml1-push-v1_mql_tcl_structure/eval.csv'
    plot_curve(axs[0], mql_struct_dir, "TCL-MQL-push")
    axs[0].legend()

    #Reach
    axs[1].set_title('MQL ML1 Reach')
    axs[1].set_xlabel('Env Steps')
    axs[1].set_ylabel('Average Test Reward')
    mql_reach_dir = '/home/simon0xzx/research/berkely_research/meta-q-learning/result/mql_ml1/metaworld-ml1-reach-v1_mql_default/eval.csv'
    plot_curve(axs[1], mql_reach_dir, "MQL-reach")

    tclmql_reach_dir = "/home/simon0xzx/research/berkely_research/meta-q-learning/result/tcl_mql_ml1/metaworld-ml1-reach-v1_mql_tcl_first/eval.csv"
    plot_curve(axs[1], tclmql_reach_dir, "TCL-MQL-reach")
    axs[1].legend()

    # PickPlace
    axs[2].set_title('MQL ML1 Pick Place')
    axs[2].set_xlabel('Env Steps')
    axs[2].set_ylabel('Average Test Reward')
    tclmql_pick_place_dir = '/home/simon0xzx/research/berkely_research/meta-q-learning/result/mql_ml1/metaworld-ml1-pick-place-v1_mql_default/eval.csv'
    plot_curve(axs[2], tclmql_pick_place_dir, "MQL-pick-place")
    tclmql_pick_place_dir = '/home/simon0xzx/research/berkely_research/meta-q-learning/result/tcl_mql_ml1/metaworld-ml1-pick-place-v1_mql_tcl_first/eval.csv'
    plot_curve(axs[2], tclmql_pick_place_dir, "TCL-MQL-pick-place")
    axs[2].legend()

    plt.show()

if __name__ == "__main__":
    report()