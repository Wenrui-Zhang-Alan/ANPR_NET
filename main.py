from Evaluation import * 
import os



print("___________ Welcome to car plate detection ___________")

# Configs

dir_path = 'evaluations'
v_path = 'video'


# Main circle

usr_input = "-1"

while(usr_input != "3"):

    usr_input = input("Choose from the following options: \n 1.Image Input \n 2.Vedio Input \n 3.QUIT \n:")

    while(usr_input != "1" and usr_input != "2" and usr_input != "3"):
        print("Invalid Input.")
        usr_input = input("Choose from the following options: \n 1.Image Input \n 2.Vedio Input \n 3.QUIT \n:")


    if usr_input == "1":

        # Finding images
        paths = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                paths.append(os.path.join(root, file))

        print("Choose from the following image to view results: ")
        for i in range(len(paths)):
            print("{}.{}".format(i,paths[i]))

        choose = input(":")

        # input check

        while(int(choose) not in range(len(paths))):
            print("Invalid Choose!")
            choose = input("Choose again:")

        p_view(paths[int(choose)])
        

    elif usr_input == "2":

        # video

        # Finding videos
        paths = []
        for root, dirs, files in os.walk(v_path):
            for file in files:
                paths.append(os.path.join(root, file))

        print("Choose from the following video to view results: ")
        for i in range(len(paths)):
            print("{}.{}".format(i,paths[i]))

        choose = input(":")

        # input check

        while(int(choose) not in range(len(paths))):
            print("Invalid Choose!")
            choose = input("Choose again:")

        additional = input("Do you wish to vis maskrcnn result?: 0 or 1 \n: ")
        while(additional != "0" and additional != "1"):
            print("Invalid Choose!")
            choose = input("Choose again:")

        v_view(paths[int(choose)],int(additional))





