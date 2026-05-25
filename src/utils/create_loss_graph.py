import matplotlib.pyplot as plt


def save_loss_curve(x_values, y_values, title, x_label, y_label, output_path):

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(False)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n{'-'*10} Graph created and saved to {output_path} {'-'*10}")