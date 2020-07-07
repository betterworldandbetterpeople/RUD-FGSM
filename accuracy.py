
label_dict = {}
with open('Predictions/labels', 'r') as file:
    for line in file:
        label_dict[line[0:27]] = int(line[27: -1]) +1

print(label_dict)

prediction_dict = {}
with open('Predictions/RUD_adv_images_resnet_v2_101_prediction', 'r') as file2:
    for line2 in file2:
        prediction_dict[line2[0:27]] = int(line2[27: -1])

print(prediction_dict)

precisioin = len(label_dict.items() & prediction_dict.items()) / len(label_dict)
print(precisioin)