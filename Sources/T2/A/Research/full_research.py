import subprocess
import time
import sys
import os
import numpy as np
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
sys.path.append('../../../')
import helper


def research_thread(num_thread, script_path, dataset_name, method, result_filename, filename_part):

    process = subprocess.Popen(['cmd', '/c', 'python', script_path
                                   , str(num_thread)
                                   , method
                                   , dataset_name
                                   , filename_part
                                   , result_filename
                                ], creationflags=subprocess.CREATE_NEW_CONSOLE)
    print(f"Task start: {num_thread}")
    start = time.perf_counter()
    process.wait()
    finish = time.perf_counter()
    print(f"Task completed: {num_thread}, time {finish - start: .2f}")
    return num_thread


def main():
    result_name = 'X_10_SCIPY-LEAST_SQUARES-TRF_BOUNDS_FULL'
    params = {
        'dataset_name': '1',
        'method': 'scipy-least_squares-trf',
        # gauss-newton / dog-leg / adam / l-bfgs
        # scipy-bfgs / scipy-l-bfgs / scipy-dog-leg
        # scipy-least_squares-dog-box / scipy-least_squares-trf
        "init_dist_x": 10,
        "init_dist_y": 0,
        "init_density_x": 101,
        "init_density_y": 1
    }
    count_threads = max(cpu_count(), 1)
    count_threads = 4
    script_path = 'one_thread_research.py'
    dataset_params = helper.get_params_dataset(params['dataset_name'])

    start = time.perf_counter()
    print(f'Start Research')
    print('params', params)
    print('Generate linspace')
    init_x = np.linspace(dataset_params['w0'] - params['init_dist_x'], dataset_params['w0'] + params['init_dist_x'], params['init_density_x'])
    init_y = np.linspace(dataset_params['w1'] - params['init_dist_y'], dataset_params['w1'] + params['init_dist_y'], params['init_density_y'])
    print('Generate meshgrid')
    X, Y = np.meshgrid(init_x, init_y)
    print('Generate combined')
    combined = list(zip(X.flatten(), Y.flatten()))
    print('Split combined')
    split_parts = np.array_split(combined, count_threads)
    filenames_parts = []
    results_filenames = []
    print('Save parts to a file')
    for i in range(len(split_parts)):
        filenames_parts.append('part ' + str(i))
        helper.save_matrix(split_parts[i], filenames_parts[i])
        results_filenames.append('result ' + str(i))

    count_tasks = params['init_density_x'] * params['init_density_y'] * dataset_params['test_count']
    print(f'Scheduling tasks - {count_tasks}')
    with ProcessPoolExecutor() as executor:
        futures = []

        for i in range(count_threads):
            executor.submit(research_thread, i, script_path, params['dataset_name'], params['method'], results_filenames[i], filenames_parts[i])

        for future in concurrent.futures.as_completed(futures):
            num_thread = future.result()

    print('Delete parts files')
    for file_path in filenames_parts:
        os.remove(file_path)

    print('Combining results')
    combined_list = []

    for file_path in results_filenames:
        combined_list.extend(helper.load_json(file_path))
        os.remove(file_path)

    print(f'Save {len(combined_list)} results')
    helper.save_json(combined_list, 'temp', indent=1)
    helper.add_result(result_name, params, 'temp')

    finish = time.perf_counter()
    print(f'It took {finish - start: .2f} second(s) to finish')


if __name__ == '__main__':
    main()
