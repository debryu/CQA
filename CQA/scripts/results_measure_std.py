import os
import ast
import math
import pandas as pd
from CQA.utils.args_utils import load_args_from_path
import matplotlib.pyplot as plt
from loguru import logger
FAST_STORAGE = os.environ["FAST"]
MODELS = [
    f"{FAST_STORAGE}/results/fixed_models/CBM/argo_random",
    f"{FAST_STORAGE}/results/fixed_models/CBM/argo_ucbf",
    f"{FAST_STORAGE}/results/fixed_models/CBM/argo_ucbf1",
    #f"{FAST_STORAGE}/results/fixed_models/CBM/argo_random_ce",
    f"{FAST_STORAGE}/results/fixed_models/CBM/cbmlite",
]

used_metrics = ['label_f1', 'f1_cal', 'rocauc', 'disentanglement']
experiments = {
                'celeba': {'labo': [],
                           'lfcbm': [],
                           'argo': [],
                           'resnetcbm': [],
                           'vlgcbm':[]}
                        ,
                'shapes3d': {   'labo': [],
                                'lfcbm': [],
                                'argo': [],
                                'resnetcbm': [],
                                'argo': [],
                                'vlgcbm':[]}
                        ,
                'dermamnist': { 'labo': [],
                                'lfcbm': [],
                                'argo': [],
                                'resnetcbm': [],
                                'vlgcbm':[]}
                        ,
                #'cub': {    'labo': [],
                #           'lfcbm': [],
                #           'resnetcbm': [],
                #           'vlgcbm':[]}
                #        ,
                }
experiments = []

for path in MODELS:
    folders = os.listdir(path)
    for folder in folders:
        if folder.endswith(".txt"):
            continue
        model = folder.split("_")[0]
        dataset = folder.split("_")[1]
        poolsize = int(folder.split("_")[-1].split("=")[-1])

        if not os.path.exists(os.path.join(path, folder, "metrics.txt")):
            logger.error(f"Failed loading {os.path.join(path, folder)}/metrics.txt")
            continue
        with open(os.path.join(path, folder, "metrics.txt"), "r") as f:
            text = f.read()
        
        args = vars(load_args_from_path(os.path.join(path,folder)))
        metrics = ast.literal_eval(text)

        # Make sure the dataset args is only the name
        # eg shapes3d_argo must become just shaped3d
        args['dataset'] = args['dataset'].split("_")[0]

        # Add all extra args
        args['poolsize'] = poolsize
        for key, value in metrics.items():
            args[key] = value
        
        if model != 'cbmlite':
            temp_name = args['argo_train']
            args['acq_fn'] = temp_name.split("-")[3]
            args['kernel'] = temp_name.split("-")[4]
            if 'loss_fn' not in args.keys():
                args['loss_fn'] = 'js'
        
        #print(metrics)
        experiments.append(args)
    
df = pd.DataFrame(experiments)

def get_metrics():
    
    return {
        "label_f1": "label_f1",
        "disentanglement": "disentanglement",
        "f1_cal": "f1_cal",
        "roc_auc": "roc_auc",
        "pr_auc": "pr_auc",
    }

def plot_metric_with_errorbars(
    models,
    model_names,
    metric,
    ylabel,
    out_path,
    colors,
):
    plt.figure(figsize=(6, 4))
    values = []
    for i,model in enumerate(models):
        try:
            grouped = model.groupby("poolsize")[metric]
            x_argo = grouped.mean().index.values
            y_argo = grouped.mean().values
            yerr_argo = grouped.std().values
            values.append({
                "x":x_argo,
                "y":y_argo,
                "yerr":yerr_argo,
                "metric":metric,
                "name": model_names[i],
            })
            plt.errorbar(
                x_argo,
                y_argo,
                yerr=yerr_argo,
                fmt="o-",
                linewidth=2,
                capsize=4,
                alpha=0.8,
                label=model_names[i],
                color= colors[i],
            )
        except:
            logger.error(f"metric {metric} not found")

    plt.xlabel("Training Pool Size")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Pool Size")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    plt.close()

def plot():
    output_dir = f"{FAST_STORAGE}/results/figs"
    for ds, color in zip(['shapes3d', 'celeba','dermamnist','cub'],['red','red','red','red']):
        df_filtered_argo_random = df[(df['dataset'] == ds) & (df['model'] == 'argo')  & (df['acq_fn'] == 'random') & (df['loss_fn'] == 'js')]
        #df_filtered_argo_random_ce = df[(df['dataset'] == ds) & (df['model'] == 'argo')  & (df['acq_fn'] == 'random') & (df['loss_fn'] == 'ce')]
        df_filtered_argo_ucbf = df[(df['dataset'] == ds) & (df['model'] == 'argo')  & (df['acq_fn'] == 'ucbf') & (df['loss_fn'] == 'js')] 
        df_filtered_argo_ucbf1 = df[(df['dataset'] == ds) & (df['model'] == 'argo')  & (df['acq_fn'] == 'ucbf1') & (df['loss_fn'] == 'js')]
        df_filtered_cbmlite = df[(df['dataset'] == ds) & (df['model'] == 'cbmlite')]
    
        for metric, ylabel in get_metrics().items():
            out_file = os.path.join(output_dir, f"{ds}_{metric}_vs_poolsize.png")
            models = [df_filtered_argo_random, df_filtered_argo_ucbf, df_filtered_argo_ucbf1, df_filtered_cbmlite]
            model_names = ['argo-random','argo-ucbf','argo-ucbf1','cbm-at']#,'argo-ucb','argo-ucb1']
            plot_metric_with_errorbars(
                models = models,
                model_names=model_names,
                metric=metric,
                ylabel=ylabel,
                out_path=out_file,
                colors = ['red','blue','green','black'],
            )

plot()

TABLE_SPEC = {
    "shapes3d": [
        ("CBM @ $100\\%$", "CBM@100"),
        ("CBM @ $0.88\\%$", "CBM@0.88"),
        ("LABO", "LABO"),
        ("LFCBM", "LFCBM"),
        ("VLGCBM", "VLGCBM"),
        ("\\method @ $420$", "argo@420"),
    ],
    "celeba": [
        ("CBM @ $100\\%$", "CBM@100"),
        ("CBM @ $1.0\\%$", "CBM@1.0"),
        ("LABO", "LABO"),
        ("LFCBM", "LFCBM"),
        ("VLGCBM", "VLGCBM"),
        ("\\method @ $390$", "argo@390"),
    ],
    "dermamnist": [
        ("CBM @ $100\\%$", "CBM@100"),
        ("CBM @ $3.2\\%$", "CBM@3.2"),
        ("LABO", "LABO"),
        ("LFCBM", "LFCBM"),
        ("VLGCBM", "VLGCBM"),
        ("\\method @ $112$", "argo@112"),
    ],
    "cub": [
        ("CBM @ $100\\%$", "CBM@100"),
        ("CBM @ $10\\%$", "CBM@10"),
        ("LABO", "LABO"),
        ("LFCBM", "LFCBM"),
        ("VLGCBM", "VLGCBM"),
        ("\\method @ $10\\%$", "argo@10"),
    ],
}

def mean_std(df, col, precision=2):
    mean = df[col].mean()
    std = df[col].std()
    return f"{mean:.{precision}f} \\pm {std:.{precision}f}"

def latex_row(df, dataset, model_key, model_label):
    if model_key.startswith("argo"):
        model, poolsize = model_key.split("@")
        sub = df[(df.dataset == dataset) & (df.model == model) & (df.poolsize == int(poolsize))]
    elif model_key.startswith("cbmlite"):
        model, poolsize = model_key.split("@")
        sub = df[(df.dataset == dataset) & (df.model == model) & (df.poolsize == int(poolsize))]
    else:
        sub = df[(df.dataset == dataset) & (df.model == model_key)]
    

    fy = mean_std(sub, "label_f1")
    mpr = mean_std(sub, "avg_macro_pr_auc")
    dis = mean_std(sub, "disentanglement")

    return (
        f"& {model_label}\n"
        f"    & ${fy}$\n"
        f"    & ${mpr}$\n"
        f"    & ${dis}$ \\\\\n"
    )

def write_latex_table(df, output): 
    with open(output, "w") as f:
        f.write("\\begin{table*}[!t]\n\\centering\n")
        f.write("\\scalebox{0.9}{\n\\begin{tabular}{llccc}\n")
        f.write("\\toprule\n")
        f.write("& {\\sc Model} & \\FY ($\\uparrow$) & \\MPR ($\\uparrow$) & \\Disent ($\\uparrow$) \\\\\n")
        f.write("\\midrule\n")

        for dataset, rows in TABLE_SPEC.items():
            dataset_macro = dataset.upper() if dataset != "dermamnist" else "DERMA"
            f.write(
                f"\\multirow{{{len(rows)}}}{{*}}{{\\rotatebox{{90}}{{\\{dataset_macro}}}}}\n"
            )

            for i, (label, key) in enumerate(rows):
                f.write(latex_row(df, dataset, key, label))
                if i == 1:
                    f.write("\\cmidrule{2-5}\n")

            f.write("\\midrule\n")

        f.write("\\bottomrule\n\\end{tabular}\n}\n")
        f.write("\\caption{Comparison of GP variations}\n")
        f.write("\\label{tab:results}\n\\end{table*}\n")

#write_latex_table(df, f"{FAST_STORAGE}/results/figs/results_table.txt")
