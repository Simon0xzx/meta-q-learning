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

def plot_curve(axs, root_dir, plot_targets, row=1, col=3):
    draw_ptr = 0
    for exp_name, sub_dir, label in plot_targets:
        subplot_axs = None
        if row == 1:
            subplot_axs = axs[draw_ptr]
        elif col == 1:
            subplot_axs = axs[draw_ptr]
        else:
            subplot_axs = axs[draw_ptr//row][draw_ptr % col]
        subplot_axs.set_title("MQL vs TCL-MQL {}".format(exp_name))
        subplot_axs.set_xlabel("Env Steps")
        subplot_axs.set_ylabel("Average Eval Reward")
        dir = path.join(root_dir, sub_dir, "eval.csv")
        timestamp, ep_reward_mean = read_csv(dir)
        subplot_axs.plot(timestamp, ep_reward_mean, lw=2, label=label)
        subplot_axs.legend()
        draw_ptr += 1


def report_namazu():
    root_dir = "/home/simon0xzx/research/berkely_research/meta-q-learning/external_data/namazu_result"
    tcl_mql_plot_target = [('reach', 'tcl_mql_ml1/metaworld-ml1-reach-v1_mql_tcl_mql','tcl_mql'),
                           ('push', 'tcl_mql_ml1/metaworld-ml1-push-v1_mql_tcl_mql','tcl_mql'),
                           ('pick-place', 'tcl_mql_ml1/metaworld-ml1-pick-place-v1_mql_tcl_mql','tcl_mql'),
                           ('reach_wall', 'tcl_mql_ml1/metaworld-ml1-reach-wall-v1_mql_tcl_mql','tcl_mql'),
                           ('sweep_into', 'tcl_mql_ml1/metaworld-ml1-sweep-into-v1_mql_tcl_mql','tcl_mql')]
    mql_plot_target = [('reach', 'mql_ml1/metaworld-ml1-reach-v1_mql_mql','mql'),
                       ('push', 'mql_ml1/metaworld-ml1-push-v1_mql_mql','mql'),
                       ('pick-place', 'mql_ml1/metaworld-ml1-pick-place-v1_mql_mql','mql'),
                       ('reach_wall', 'mql_ml1/metaworld-ml1-reach-wall-v1_mql_mql','mql'),
                       ('sweep_into', 'mql_ml1/metaworld-ml1-sweep-into-v1_mql_mql','mql')]
    row, col = 1, 5
    fig, axs = plt.subplots(row, col)
    plot_curve(axs, root_dir, tcl_mql_plot_target, row=row, col=col)
    plot_curve(axs, root_dir, mql_plot_target, row=row, col=col)
    plt.show()

def report_jormungandr():
    root_dir = "/Users/simon/Research/meta-q-learning/external_data/jormungandr_result/tcl_mql_ml1"

    namazu_dir = "/Users/simon/Research/meta-q-learning/external_data/namazu_result"
    tcl_mql_itr1_target = [('reach', 'tcl_mql_ml1/metaworld-ml1-reach-v1_mql_tcl_mql','iter1'),
                           ('push', 'tcl_mql_ml1/metaworld-ml1-push-v1_mql_tcl_mql','iter1'),
                           ('pick-place', 'tcl_mql_ml1/metaworld-ml1-pick-place-v1_mql_tcl_mql','iter1')]
    mql_plot_target = [('reach', 'mql_ml1/metaworld-ml1-reach-v1_mql_mql','mql'),
                       ('push', 'mql_ml1/metaworld-ml1-push-v1_mql_mql','mql'),
                       ('pick-place', 'mql_ml1/metaworld-ml1-pick-place-v1_mql_mql','mql')]
    tcl_mql_itr2_target = [('reach', 'metaworld-ml1-reach-v1_mql_tcl_mql2', 'iter2'),
                           ('push', 'metaworld-ml1-push-v1_mql_tcl_mql2', 'iter2'),
                           ('pick-place', 'metaworld-ml1-pick-place-v1_mql_tcl_mql2', 'iter2')]
    tcl_mql_itr3_target = [('reach', 'metaworld-ml1-reach-v1_mql_tcl_mql3', 'iter3'),
                           ('push', 'metaworld-ml1-push-v1_mql_tcl_mql3', 'iter3'),
                           ('pick-place', 'metaworld-ml1-pick-place-v1_mql_tcl_mql3', 'iter3')]
    tcl_mql_itr4_target = [('reach', 'metaworld-ml1-reach-v1_mql_tcl_mql4', 'iter4'),
                           ('push', 'metaworld-ml1-push-v1_mql_tcl_mql4', 'iter4'),
                           ('pick-place', 'metaworld-ml1-pick-place-v1_mql_tcl_mql4', 'iter4')]
    tcl_mql_itr5_target = [('reach', 'metaworld-ml1-reach-v1_mql_tcl_mql5', 'iter5'),
                           ('push', 'metaworld-ml1-push-v1_mql_tcl_mql5', 'iter5'),
                           ('pick-place', 'metaworld-ml1-pick-place-v1_mql_tcl_mql5', 'iter5')]
    tcl_mql_itr6_target = [('reach', 'tcl_mql_ml1/metaworld-ml1-reach-v1_mql_tcl_mql6', 'iter6'),
                           ('push', 'tcl_mql_ml1/metaworld-ml1-push-v1_mql_tcl_mql6', 'iter6'),
                           ('pick-place', 'tcl_mql_ml1/metaworld-ml1-pick-place-v1_mql_tcl_mql6', 'iter6'),]
    row, col = 1, 3
    fig, axs = plt.subplots(row, col)
    # plot_curve(axs, namazu_dir, mql_plot_target, row=row, col=col)
    # plot_curve(axs, namazu_dir, tcl_mql_itr1_target, row=row, col=col)
    plot_curve(axs, root_dir, tcl_mql_itr2_target, row=row, col=col)
    plot_curve(axs, root_dir, tcl_mql_itr3_target, row=row, col=col)
    plot_curve(axs, root_dir, tcl_mql_itr4_target, row=row, col=col)
    plot_curve(axs, root_dir, tcl_mql_itr5_target, row=row, col=col)
    plot_curve(axs, namazu_dir, tcl_mql_itr6_target, row=row, col=col)



    plt.show()

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
    # report()
    # report_namazu()
    report_jormungandr()