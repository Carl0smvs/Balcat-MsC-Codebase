import dataset_load as utils
import matplotlib.pyplot as plt
import numpy as np

EXPERIMENT_NAME = '06-05 (Original model generic bug fixed 200 epochs)'
EXPERIMENT_DIR = f'../experiments/{EXPERIMENT_NAME}/'
CALIBERS_TO_PLOT = ["6-35MM", "7-65MM", "9MM"]

for CALIBER_TO_PLOT in CALIBERS_TO_PLOT:

    EVALUATION_FILES_DIR = f'{EXPERIMENT_DIR}{CALIBER_TO_PLOT}/evaluation'

    eval_files = utils.list_files(EVALUATION_FILES_DIR)

    data_per_model = dict()
    for eval_file in eval_files:
        # Each file has the test metrics for each model
        f = open(eval_file, 'r')
        lines = f.readlines()
        f.close()

        model_name = eval_file.split('/')[-1].split('-')[0]
        data_per_model[model_name] = dict()

        for i, line in enumerate(lines):
            if line.startswith('Overall Accuracy'):
                data_per_model[model_name]['accuracy'] = float(line.split(':')[-1].strip())
            if line.startswith('Top 3 Accuracy'):
                data_per_model[model_name]['top-3-accuracy'] = float(line.split(':')[-1].strip())
            if line.startswith('F1 Score'):
                data_per_model[model_name]['f1-score'] = float(line.split(':')[-1].strip())
            if line.startswith('Overall Loss'):
                data_per_model[model_name]['loss'] = float(line.split(':')[-1].strip())

            if 5 <= i <= 44:
                j = i - 5
                class_name = lines[i - j % 4].split('metrics')[0].strip()
                if 'class_names' not in data_per_model[model_name].keys():
                    data_per_model[model_name]['class_names'] = []
                if j % 4 == 0:
                    data_per_model[model_name]['class_names'].append(class_name)
                    data_per_model[model_name][class_name] = []
                else:
                    metric = line.split(':')[-1].strip()
                    if metric == 'nan':
                        metric = '0'
                    data_per_model[model_name][class_name].append(float(metric))

    if len(data_per_model) == 0:
        continue

    num_models = len(data_per_model)
    num_classes = 10

    models = [model_name for model_name in data_per_model.keys()]
    accuracies = [data_per_model[model_name]['accuracy'] for model_name in models]
    top_3_accuracies = [data_per_model[model_name]['top-3-accuracy'] for model_name in models]
    f1_scores = [data_per_model[model_name]['f1-score'] for model_name in models]
    losses = [data_per_model[model_name]['loss'] for model_name in models]
    # Overall Metrics Plot

    fig, axs = plt.subplots(1, 4, figsize=(45, 5))
    axs[0].bar(models, accuracies)
    axs[0].set_xlabel("Model Type")
    axs[0].set_ylabel("Accuracy")
    #axs[0].yticks(np.arange(1, step=0.1))
    axs[0].set_title("Test Categorical Accuracy per Model Type")

    axs[1].bar(models, top_3_accuracies)
    axs[1].set_xlabel("Model Type")
    axs[1].set_ylabel("Top 3 Accuracy")
    axs[1].set_title("Test Top 3 Accuracy per Model Type")

    axs[2].bar(models, f1_scores)
    axs[2].set_xlabel("Model Type")
    axs[2].set_ylabel("F1 Score")
    axs[2].set_title("Test F1 Score per Model Type")

    axs[3].bar(models, losses)
    axs[3].set_xlabel("Model Type")
    axs[3].set_ylabel("Loss")
    axs[3].set_title("Test Loss per Model Type")

    plt.savefig(f'{EXPERIMENT_DIR}{CALIBER_TO_PLOT}/overall-metrics-comparison.png')
    plt.cla()
    plt.clf()

    # Aggregated Overall Metrics Plot
    x = np.arange(3)  # the number of metrics as labels
    width = 0.35  # the width of the bars

    xception_metrics = [data_per_model['XCEPTION']['accuracy'], data_per_model['XCEPTION']['top-3-accuracy'], data_per_model['XCEPTION']['f1-score']]
    vgg_metrics = [data_per_model['VGG_16']['accuracy'], data_per_model['VGG_16']['top-3-accuracy'], data_per_model['VGG_16']['f1-score']]
    inception_metrics = [data_per_model['INCEPTION_V3']['accuracy'], data_per_model['INCEPTION_V3']['top-3-accuracy'], data_per_model['INCEPTION_V3']['f1-score']]
    efficient_metrics = [data_per_model['EFFICIENT_NET_B0']['accuracy'], data_per_model['EFFICIENT_NET_B0']['top-3-accuracy'], data_per_model['EFFICIENT_NET_B0']['f1-score']]
    dense_metrics = [data_per_model['DENSE_NET_121']['accuracy'], data_per_model['DENSE_NET_121']['top-3-accuracy'], data_per_model['DENSE_NET_121']['f1-score']]

    fig, ax = plt.subplots()
    # We want the measurements of the 5 models for each of the 3 metrics
    rectsXception = ax.bar(x - width, xception_metrics, width / 2, label='Xception')
    rectsVgg = ax.bar(x - width / 2, vgg_metrics, width / 2, label='VGG 16')
    rectsInception = ax.bar(x, inception_metrics, width / 2, label='Inception V3')
    rectsEfficient = ax.bar(x + width / 2, efficient_metrics, width / 2, label='Efficient Net B0')
    rectsDense = ax.bar(x + width, dense_metrics, width / 2, label='Dense Net 121')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Aggregated metrics comparison between every model')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Top-3 Accuracy', 'F1-score'])
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = round(rect.get_height(), 2)
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rectsXception)
    autolabel(rectsVgg)
    autolabel(rectsInception)
    autolabel(rectsEfficient)
    autolabel(rectsDense)

    fig.tight_layout()
    plt.savefig(f'{EXPERIMENT_DIR}{CALIBER_TO_PLOT}/aggregated-metrics-comparison.png')
    plt.cla()
    plt.clf()

    model_metrics = [[data_per_model[model][class_name]
                         for class_name in data_per_model[model]['class_names']]
                        for model in models]

    # Precision per class plot
    fig, ax = plt.subplots(figsize=(50, 10))
    x = np.arange(30, step=3) # the number of classes
    bar_width = 0.5

    for i, model in enumerate(models):
        ax.bar(x + i * bar_width, np.array(model_metrics)[i, :, 0], width=bar_width, label=model)

    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels(data_per_model[models[0]]['class_names'])

    ax.legend()

    # Axis styling.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    # Add axis and chart labels.
    ax.set_xlabel('Firearm Model', labelpad=15)
    ax.set_ylabel('Precision', labelpad=15)
    ax.set_title(f'Precision score by Model for each {CALIBER_TO_PLOT} class', pad=15)
    plt.savefig(f'{EXPERIMENT_DIR}{CALIBER_TO_PLOT}/class-precision-comparison.png')
    plt.cla()
    plt.clf()

    # Recall per class plot
    fig, ax = plt.subplots(figsize=(50, 10))
    x = np.arange(30, step=3) # the number of classes
    bar_width = 0.5

    for i, model in enumerate(models):
        ax.bar(x + i * bar_width, np.array(model_metrics)[i, :, 1], width=bar_width, label=model)

    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels(data_per_model[models[0]]['class_names'])

    ax.legend()

    # Axis styling.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    # Add axis and chart labels.
    ax.set_xlabel('Firearm Model', labelpad=15)
    ax.set_ylabel('Recall', labelpad=15)
    ax.set_title(f'Recall score by Model for each {CALIBER_TO_PLOT} class', pad=15)
    plt.savefig(f'{EXPERIMENT_DIR}{CALIBER_TO_PLOT}/class-recall-comparison.png')
    plt.cla()
    plt.clf()

    # F1 Score per class plot
    fig, ax = plt.subplots(figsize=(50, 10))
    x = np.arange(30, step=3) # the number of classes
    bar_width = 0.5

    for i, model in enumerate(models):
        ax.bar(x + i * bar_width, np.array(model_metrics)[i, :, 2], width=bar_width, label=model)

    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels(data_per_model[models[0]]['class_names'])

    ax.legend()

    # Axis styling.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    # Add axis and chart labels.
    ax.set_xlabel('Firearm Model', labelpad=15)
    ax.set_ylabel('F1-Score', labelpad=15)
    ax.set_title(f'F1-Score by Model for each {CALIBER_TO_PLOT} class', pad=15)
    plt.savefig(f'{EXPERIMENT_DIR}{CALIBER_TO_PLOT}/class-f1-score-comparison.png')
    plt.cla()
    plt.clf()
