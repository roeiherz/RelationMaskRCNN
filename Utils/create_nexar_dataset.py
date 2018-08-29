import argparse
import os
import re
import pandas as pd
import numpy as np

LANE_CSV_FILE = "data.csv"

INDEX = 'data/Accidents/index_tmp.csv'
IMAGE_PATH = "/data/Incidents/Images"


def get_frames_num(df_index, incident, output_path):
    """
    This function will number of frames
    :param output_path:
    :param incident:
    :param df_index:
    :return: pb_data mapping dict
    """

    # Get video base name for dumping the file
    query = df_index[df_index['incident id'] == incident]['video link']
    video_base_name = os.path.basename(list(query)[0]).split('.')[0]  # without the .mov

    # without .mov file
    if os.path.exists(os.path.join(output_path, video_base_name)):
        path = os.path.join(output_path, video_base_name, 'images')
    else:
        # with .mov file
        video_base_name_mov = os.path.basename(list(query)[0])
        path = os.path.join(output_path, video_base_name_mov, 'images')

    files = os.listdir(path)
    return len(files)


def get_lanes_annot(path):
    with open(path, 'rb') as myfile:
        rows = myfile.readlines()
        contents = [x.strip() for x in rows]

        data = [content.replace(' ', '').replace('\"', '') for content in contents]
        data = [[d[:d.find(',')], d[d.find(',') + 1:]] for d in data]
        incidents = [tup[0] for tup in data]
        times_lst = []
        values_lst = []
        for tup in data:
            try:
                if len(tup) > 1:

                    # Get Times in milliseconds
                    start_time_annot = tup[1].find(':')
                    end_time_annot = tup[1].find('v')
                    times = tup[1][start_time_annot + 1: end_time_annot]
                    times = [float(t) for t in times.split(',')]

                    # Get Values of the lanes
                    start_val_annot = tup[1][end_time_annot:].find(':')
                    start_val_annot += end_time_annot
                    values = tup[1][start_val_annot + 1:]
                    values = [int(re.search(r'\d+', t).group()) - 1 if "Lane" in t else -1
                              for t in values.split(',')]

                    if len(times) != len(values):
                        print("Problem between the times and values")

                    # Append
                    times_lst.append(times)
                    values_lst.append(values)

            except Exception as e:
                print("Error in incident {} with {}".format(tup[0], e))

    # Define the cols for the DataFrame
    dataframe_labels = ["Incidents", "Times", "Lanes"]
    data = {"Incidents": incidents, "Times": times_lst, "Lanes": values_lst}
    # Define DataFrame
    df_lane_annot = pd.DataFrame(data=data, columns=dataframe_labels)

    return df_lane_annot


def save_df_gt_labels(output_path, data_path_pd, label_pd, frames_ids):
    """
    This function saves the data frame with the prediction of the labels GT
    :param output_path: output path to save
    :return:
    """
    # Define the cols for the DataFrame
    dataframe_labels = ["Frame_ID", "Image_Path", "Lane", "Lane_Label"]
    data = {"Image_Path": data_path_pd, "Lane": label_pd, "Lane_Label": label_pd, "Frame_ID": frames_ids}
    # Define DataFrame
    df = pd.DataFrame(data=data, columns=dataframe_labels)
    # Save Dataframe
    df.to_csv(output_path)


if __name__ == '__main__':
    
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', help='local variable for debugging', action='store', default=False)
    parser.add_argument('--index', help='index csv file', action='store', default=INDEX)
    parser.add_argument('--image', help='directory of parsered images', action='store', default=IMAGE_PATH)
    # parser.add_argument('--annot', help='annotations csv file', action='store', default=ANNON)
    args = parser.parse_args()

    # Use Local params
    if args.local:
        # args.annot = "/Users/roeiherzig/Datasets/Incidents/raw_annotations.csv"
        args.index = "/Users/roeiherzig/Datasets/Accidents/index_tmp.csv"
        args.image = "/Users/roeiherzig/Datasets/Accidents/Images"

    # Check directory exists
    if not os.path.exists(args.image):
        print('Can not find videos directory: {}'.format(args.input))
        exit(-1)

    # Load Index Dataframe
    df_index = pd.read_csv(args.index)
    df_index = df_index[df_index.Valid_Accident == "yes"]
    # df_index = df_lanes_annot[df_lanes_annot.Incidents == "a1cecccbb9d1f98445658ccf4b841ea6"]

    # Run over the df lanes
    for index, row in df_index.iterrows():
        try:
            incident = row.Incident
            start_frameid = row.Accident_StarTime
            end_frameid = row.Accident_EndTime
            print("Incident {} is been processing".format(incident))

            img_dir = "{}/incident-{}".format(args.image, incident)

            if not os.path.exists(img_dir):
                print("Dir {} not exists".format(img_dir))
                continue

            # Get files
            files = os.listdir(img_dir)
            path_lst = [os.path.join(img_dir, fl) for fl in files]
            incident_lst = [incident] * len(path_lst)

            # Get frames
            start_time = float(df_index[df_index['incident id'] == incident]['start epoch [sec]'].item())
            end_time = float(df_index[df_index['incident id'] == incident]['end epoch [sec]'].item())

            frame_ids = []
            lanes_label_lst = []
            img_path_lst = []


            # Get video link path
            query = df_index[df_index['Incident'] == incident]['Link']
            link = list(query)[0]
            video_base_name = link[link.find("incident") + 9:]  # without the .mov

            # Get frames
            start_time = float(df_index[df_index['incident id'] == incident]['start epoch [sec]'].item())
            end_time = float(df_index[df_index['incident id'] == incident]['end epoch [sec]'].item())
            frames_num = get_frames_num(df_index, incident, args.output)
            # frames_num = 1266
            tms_per_incident_arr = np.linspace(start_time, end_time, frames_num)

            # Times in epoch
            epoch_times = [time + start_time for time in times]
            if lanes[-1] != -1 and epoch_times[-1] < end_time:
                # Append the last lane
                lanes.append(lanes[-1])
                # Append the end time
                epoch_times.append(end_time)
            # Find frames per epoch times
            frames_per_times = [(np.abs(tms_per_incident_arr - time)).argmin() for time in epoch_times]

            # Find consecutive element in the lanes
            lanes_consecutive = np.ediff1d(lanes)

            frame_ids = []
            lanes_label_lst = []
            img_path_lst = []

            for ind in range(len(frames_per_times) - 1):

                if lanes[ind] == -1:
                    continue

                for i in range(frames_per_times[ind], frames_per_times[ind + 1]):
                    frame_ids.append(i)
                    lanes_label_lst.append(lanes[ind])
                    # Image Path
                    image_path = os.path.join(video_base_name, "{0:06d}.jpg".format(i))
                    img_path_lst.append(image_path)

            # Finished epoch Save csv GT for lanes
            csv_file = os.path.join(args.output, video_base_name, LANE_CSV_FILE)
            save_df_gt_labels(csv_file, img_path_lst, lanes_label_lst, frame_ids)
            print("File {} saved".format(csv_file))

        except Exception as e:
            print("Error in incident {} with {}".format(incident, e))
