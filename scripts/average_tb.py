from tensorboard.backend.event_processing import event_accumulator
from statistics import mean
import os 
import csv


def read_file(my_file):
	ea = event_accumulator.EventAccumulator(my_file,
			 size_guidance={ 
			event_accumulator.COMPRESSED_HISTOGRAMS: 500,
			event_accumulator.IMAGES: 4,
			event_accumulator.AUDIO: 4,
			event_accumulator.SCALARS: 0, 
			event_accumulator.HISTOGRAMS: 1,
		})

	ea.Reload()
	print(ea.Tags())
	print()

	return ea


def get_metrics(subject_, ea_, my_file):
	### Validation accuracy
	val_acc = []
	for i in ea_.Scalars('epoch_acc'):
		val_acc.append(i[2])

	### Validation loss
	val_loss = []
	for i in ea_.Scalars('epoch_loss'):
		val_loss.append(i[2])

	if "train" in my_file:
		"""
		print("max train accuracy:", max(val_acc))
		print("min train accuracy:", min(val_acc))
		print("mean train accuraccy:", mean(val_acc))

		print()

		print("max train loss:", max(val_loss))
		print("min train loss:", min(val_loss))
		print("mean train loss:", mean(val_loss))

		print()
		"""

		my_dict = {
		"Subject": subject_,
		"max train accuracy": max(val_acc), 
		"min train accuracy": min(val_acc),
		"mean train accuraccy": mean(val_acc),
		"max train loss": max(val_loss), 
		"min train loss": min(val_loss),
		"mean train loss": mean(val_loss)
		}

		return my_dict


	elif "validation" in my_file:
		"""
		print("max validation accuracy:", max(val_acc))
		print("min validation accuracy:", min(val_acc))
		print("mean validation accuraccy:", mean(val_acc))

		print()

		print("max validation loss:", max(val_loss))
		print("min validation loss:", min(val_loss))
		print("mean validation loss:", mean(val_loss))

		print()
		"""

		my_dict = {
		"Subject": subject_,
		"max validation accuracy": max(val_acc), 
		"min validation accuracy": min(val_acc),
		"mean validation accuraccy": mean(val_acc),
		"max validation loss": max(val_loss), 
		"min validation loss": min(val_loss),
		"mean validation loss": mean(val_loss)
		}

		return my_dict


	return


def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def run_all(root_dir_):	
	subject_files = []
	for file in os.listdir(root_dir_):
		if "P" in file:
			subject_files.append(file)

	#print(subject_files)
	#print()
	dict_list = []
	for subject in subject_files:
		train_dir = root_dir_ + "/" + subject + "/train"
		val_dir = root_dir_ + "/" + subject + "/validation"

		for file in os.listdir(train_dir):
			if file.endswith('.v2'):
				train_file = train_dir + "/" + file

		for file in os.listdir(val_dir):
			if file.endswith('.v2'):
				val_file = val_dir + "/" + file

		print("Subject:", subject)
		train_dict = get_metrics(subject, read_file(train_file), train_file)
		val_dict = get_metrics(subject, read_file(val_file), val_file)

		
		csv_dict = Merge(train_dict,val_dict)
		dict_list.append(csv_dict)
		
		### write the dictionary to csv
		field_names = [key for key, val in csv_dict.items()]
	

	with open('13_subject_models_0_25.csv', 'a') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=field_names)
		writer.writeheader()
		writer.writerows(dict_list)

	return



if __name__ == "__main__":

	root_dir = "/Users/malihalac/Desktop/BCI_Hackathon/tensorboard/13_subject_models/0.25"
	#run_all(root_dir)
	read_file("/Users/malihalac/Desktop/BCI_Hackathon/tensorboard/subject_models/tensorboard_dir/P01/validation/events.out.tfevents.1627324310.0208a1262cbd.73.12064.v2")





