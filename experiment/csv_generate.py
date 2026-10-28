import csv
import os


def find_max_weighted_sum_index(acc_list, nmi_list, pur_list, ari_list, acc_weight, nmi_weight, pur_weight, ari_weight):
    max_sum = float('-inf')
    max_index = -1

    for i, (acc, nmi, pur, ari) in enumerate(zip(acc_list, nmi_list, pur_list, ari_list)):
        current_sum = acc * acc_weight + nmi * nmi_weight + pur * pur_weight + ari * ari_weight
        if current_sum > max_sum:
            max_sum = current_sum
            max_index = i

    return max_index


def save_lists_to_file(acc_list, nmi_list, pur_list, ari_list, loss_list, data_name, data_rate, Valid_check_num):

    csv_path = f'3.csv'
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    file_path = os.path.join(csv_path, f'{data_name}_{data_rate}.csv')

    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(['epoch', 'acc', 'nmi', 'pur', 'ari', 'loss'])

        epoch = 1*Valid_check_num
        for acc, nmi, pur, ari, loss in zip(acc_list, nmi_list, pur_list, ari_list, loss_list):
            csvwriter.writerow([epoch, acc, nmi, pur, ari, loss])
            epoch += 1*Valid_check_num

    print(f'Metrics have been saved at {file_path}')


def find_max_last_element_index(acc_l):

    max_index = 0
    max_value = acc_l[0][-1]  

    for index, sublist in enumerate(acc_l):
        if sublist[-1] > max_value:  
            max_value = sublist[-1]  
            max_index = index  

    return max_index

