import matplotlib.pyplot as plt

def plot_statistic(result_dict, title, save_name):
    # Extract methods and values for plotting
    methods = list(result_dict.keys())
    values = list(result_dict.values())

    plt.bar(methods, values)
    plt.xticks(rotation=10)
    plt.xlabel('Algorithm')
    plt.ylabel('Time to Convergence (0.99)')
    plt.title(title)

    plt.savefig(save_name, dpi=400)
    plt.close()
