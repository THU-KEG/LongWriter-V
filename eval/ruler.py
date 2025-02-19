import os
from tqdm import tqdm
import pandas as pd
from inference.local.qwen2_vl import Qwen2VL
from utils import count_words
import matplotlib.pyplot as plt
import numpy as np

def plot(model_files, save_path):
    plt.figure(figsize=(10, 8))

    # Define line styles and colors similar to the reference image
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    linestyles = ['-', '-', '-', '-', '-', '-', '-', '-']
    markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']

    for i, (model_name, file_path) in enumerate(model_files.items()):
        if file_path:  # skip empty file paths
            df = pd.read_excel(file_path)
            
            # calculate average length for each required length
            avg_lengths = df.groupby('L')['length'].mean().reset_index()
            
            # sort by required length (l) for proper line plotting
            avg_lengths = avg_lengths.sort_values('L')
            
            # plot the line for this model using averages with consistent styling
            plt.plot(avg_lengths['L'], avg_lengths['length'], 
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)],
                    marker=markers[i % len(markers)],
                    linewidth=4,
                    markersize=12,
                    label=model_name)

    # add diagonal reference line
    max_len = max([pd.read_excel(f)['L'].max() for f in model_files.values() if f])
    diag_line = np.linspace(400, max_len, 100)  # Start from 400
    plt.plot(diag_line, diag_line, 'k--', alpha=0.7)

    # customize the plot with larger font sizes
    plt.xlabel('Required Length', fontsize=20, fontweight='bold')
    plt.ylabel('Output Length', fontsize=20, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=20)

    # set axis scales and ticks
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(400, max_len*1.1)  # Set x-axis limits to start from 400
    plt.ylim(0, max_len*1.1)  # Set y-axis limits to start from 0
    plt.xticks([500, 1000, 2000, 4000], ['500', '1k', '2k', '4k'])
    plt.yticks([500, 1000, 2000, 4000], ['500', '1k', '2k', '4k'])

    plt.legend(loc='upper left', fontsize=18)
    
    plt.tight_layout()
    
    # Save as PDF vector format
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

def get_image_path(idx, row):
    """Get image path(s) from row data"""
    image_base_path = os.getenv('IMAGE_BASE_PATH', 'data/LongWriter_V_Ruler')
    image_path = row['image_path']
    if isinstance(image_path, str) and image_path.startswith('[') and image_path.endswith(']'):
        image_paths = eval(image_path)
        return [os.path.join(image_base_path, p) for p in image_paths]
    else:
        return [os.path.join(image_base_path, str(idx) + '.jpg')]

def eval_ruler(model_path, output_path):
    data_path = os.getenv('DATA_PATH', 'data/LongWriter_V_Ruler.xlsx')
    if os.path.exists(output_path):
        df = pd.read_excel(output_path)
    else:
        df = pd.read_excel(data_path)

    model = Qwen2VL(model_path)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if not pd.isna(row['prediction']):
            continue
        question = row['question']
        image_paths = get_image_path(idx, row)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    }
                ] + [
                    {
                        "type": "image",
                        "image": "file://" + p
                    }
                    for p in image_paths
                ]
            }
        ]
        sample_params = {
            "temperature": 0.9,
            "max_tokens": 8192,
            "top_p": 0.95,
            "top_k": 0,
        }
        sample_params["top_k"] = -1
        res = model.inference_vllm(messages, **sample_params)[0]
        print(res)
        df.loc[idx, 'prediction'] = res
        df.loc[idx, 'length'] = count_words(res)

        df.to_excel(output_path, index=False)
