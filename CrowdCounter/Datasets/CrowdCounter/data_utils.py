import json
import glob

def load_counterspeech_data(main_path, left_path,remove_workers):
    ''' the files are located in the two locations one is the main location
    other is the annotations collected after the main round
    remove workers is a list which tells which workers' annotation to not consider'''

    ### final dataset for counterspeech
    counterspeech_data={}

    all_files_main=glob.glob(main_path)
    all_files_left=glob.glob(left_path)
    all_files =  all_files_main + all_files_left
    for file in all_files:
        with open(file,'r') as f:
            dict_batch = json.load(f)
            print(file, len(dict_batch))
    
            for key in dict_batch:
                try:
                    temp = counterspeech_data[key]                      
                    for key_counterspeech in dict_batch[key]['counterspeech_post'].keys():
                        if(('Left' in file) and (key_counterspeech in remove_workers)):
                            print("pp")
                            continue

                        counterspeech_data[key]['counterspeech_post'][key_counterspeech]= dict_batch[key]['counterspeech_post'][key_counterspeech]

                except KeyError:
                    counterspeech_data[key]=dict_batch[key]
                

    #counte total pairs

    count_pairs=0
    for key in counterspeech_data.keys():
        count_pairs+=len(counterspeech_data[key]['counterspeech_post'].keys())
    
    print("Total number of counterspeech pairs: ", count_pairs)

    ### remove the counterspeech posts which are empty
    count_counterspeech_removed = 0
    for key in counterspeech_data:
        key_to_remove=[]
        for key_counterspeech in counterspeech_data[key]['counterspeech_post'].keys():
            if (counterspeech_data[key]['counterspeech_post'][key_counterspeech]['counterspeech']=='' or counterspeech_data[key]['counterspeech_post'][key_counterspeech]['counterspeech']==[]):
                key_to_remove.append(key_counterspeech)
        count_counterspeech_removed+=len(key_to_remove)
        for key_remove in key_to_remove:
            counterspeech_data[key]['counterspeech_post'].pop(key_remove)

    print("Number of counterspeech pairs removed: ", count_counterspeech_removed)
    
    
    print("Total pairs now: ", count_pairs - count_counterspeech_removed)
    
    return counterspeech_data

    

    