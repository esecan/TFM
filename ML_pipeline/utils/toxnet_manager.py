import os

folder = 'F:/web-scraping/chemidplus/informative_files/'
endpoint = ['rat', 'LD50', 'skin']

for filename in os.listdir(folder):
    with open('rat_skin_acute_toxicity.txt', 'a') as output_file:
        f = open(folder + filename, "r")
        tox_info = f.readline().split()
        for n in range(len(tox_info)):
            if tox_info[n] == endpoint[0]:
                if tox_info[n + 1] == endpoint[1]:
                    if tox_info[n + 2] == endpoint[2]:
                        info = [filename.split('.')[0], tox_info[n], tox_info[n+1], tox_info[n+2], tox_info[n+3], tox_info[n+4], tox_info[n+5]]
                        info_str = '\t'.join(info)
                        print(info_str)
                        output_file.write(info_str + '\n')
