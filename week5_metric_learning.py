from src.metric_learning.build_dataset import generate_metric_learning_database, plot_random_crops
from src.metric_learning.train import train
from src.metric_learning.embeddings import extract_embeddings, plot_embeddings
from pathlib import Path
from week5_task2 import task2

"""
generate_metric_learning_database(dataset_path=str(Path.joinpath(Path(__file__).parent, './data/AICity_track_data/train').absolute()),
                                  train_seq=['S01', 'S04'],
                                  test_seq=['S03'],
                                  img_size=128)
"""

#plot_random_crops()

epochs = 30
ps = [10, 20, 30]
ks = [20, 40, 60, 80]
margins = [0.1, 0.2, 0.5, 0.8, 1]

for p in ps:
    for k in ks:
        for margin in margins:
            print(p, k, margin)
            train(output_path=str(Path.joinpath(Path(__file__).parent, './checkpoints')),
                epochs=epochs,
                lr=0.0001,
                p=p,
                k=k,
                margin=margin,
                num_workers=12)

            compute_video = False
            summary_results = f'./results/week5/summary.txt'
            idfs = []

            param_space = [
                {
                'sequence': 'S03',
                'checkpoint': f'./checkpoints/epoch_{epochs-1}.pth',
                'method': 'histogram',
                'metric': 'euclidean',
                'thresh': 0.33,
                'num_bins': 32,
                },
                {
                'sequence': 'S03',
                'checkpoint': f'./checkpoints/epoch_{epochs-1}.pth',
                'method': 'histogram',
                'metric': 'euclidean',
                'thresh': 0.33,
                'num_bins': 64,
                },
                {
                'sequence': 'S03',
                'checkpoint': f'./checkpoints/epoch_{epochs-1}.pth',
                'method': 'histogram',
                'metric': 'histogram_correl',
                'thresh': 0.33,  # need to supervise first
                'num_bins': 32,
                },
                {
                'sequence': 'S03',
                'checkpoint': f'./checkpoints/epoch_{epochs-1}.pth',
                'method': 'histogram',
                'metric': 'histogram_correl',
                'thresh': 0.33,  # need to supervise first
                'num_bins': 64,
                },
                {
                'sequence': 'S03',
                'checkpoint': f'./checkpoints/epoch_{epochs-1}.pth',
                'method': 'histogram',
                'metric': 'histogram_hellinger',
                'thresh': 0.33,  # need to supervise first
                'num_bins': 32,
                },
                {
                'sequence': 'S03',
                'checkpoint': f'./checkpoints/epoch_{epochs-1}.pth',
                'method': 'metric',
                'metric': 'euclidean',
                'thresh': 15,
                'num_bins': -1,
                },
                {
                'sequence': 'S03',
                'checkpoint': f'./checkpoints/epoch_{epochs-1}.pth',
                'method': 'metric',
                'metric': 'cosine',
                'thresh': 0.2,
                'num_bins': -1,
                },
            ]
            for params in param_space:
                _, summary = task2(**params)
                print(f' IDF1 {summary["idf1"]["metrics"] * 100}')
                idfs.append(summary["idf1"]["metrics"] * 100)
                with open(summary_results, 'a') as f:
                    text = ', '.join(f'{key}: {value}' for key, value in params.items())
                    text += f'=> {summary["idf1"]["metrics"] * 100}\n'
                    f.write(text)


            best_params = param_space[idfs.index(max(idfs))]
            print(f' Best configuration is {best_params}')
            exit(1)