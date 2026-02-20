


from torch.utils.data import Dataset

import torch
import random
import numpy as np

import numpy as np






class SlipNoise_Welch_CustomDataset(Dataset):
        
    def __init__(self, data_split, set_mode,  clip_len, cfg):


        self.cfg = cfg
 
        self.clip_len = clip_len




        self.synchro_offset_torques = self.cfg.dataset.base.synchro_offset_torques
            

        #conf
        self.minfreq = self.cfg.net_extras.minfreq # 2 is 100Hz+
        self.maxfreq = self.cfg.net_extras.maxfreq # 22 is ~ 1kHz , max 129 = 5kHz

        self.set_mode = set_mode

        
        self.dataset = data_split
        self.synchronize_spectrogram_causally(self.dataset) #  offset on fft to synchronize signals if self.cfg.dataset.base.num_welch_pad
        self.num_samples = len(self.dataset)

 






    def synchronize_spectrogram_causally(self, dataset):



        for seg_dict in dataset : 

            # num_welch_pad = 10 # legacy  for fft is 1024 pze datapoints, 
            num_welch_pad =  self.cfg.dataset.base.num_welch_pad 

            if num_welch_pad : 
                

                mem_before = seg_dict['tokyo_fft'].nbytes 
                welch = seg_dict['tokyo_fft'] 
                welch_pad = np.zeros((welch.shape[0], num_welch_pad), dtype=welch.dtype)
                welch = np.concatenate([welch_pad, welch,], axis=1)[:, :-num_welch_pad]
                welch = np.ascontiguousarray(welch)
                seg_dict['tokyo_fft_synchro_padded'] = welch
                mem_after = seg_dict['tokyo_fft_synchro_padded'].nbytes 
                
                if mem_before != mem_after : 
                    print("error mem diff")
                    exit()

            else : 
                
                seg_dict['tokyo_fft_synchro_padded'] = seg_dict['tokyo_fft']




    def __len__(self):
        return len(self.dataset)



    def __getitem__(self, idx):


        seg_dict = self.dataset[idx]
        
        assert seg_dict["contact_mode"] in ["air", "slip", "noslip", "RSC", "RSCf0", "RSCf2", "noslipRSC"], f"contact mode  : {seg_dict['contact_mode'] }"
 
        

        welch = seg_dict['tokyo_fft_synchro_padded']
        slip_labels_inwelch = seg_dict['slip_labels_inwelch'] # a list of (start, end) pairs
        dense_slip_label = seg_dict['dense_slip_label']
        dense_noise_label = seg_dict["dense_noise_label"]



        #  offset on bench ground truth to synchronize  signals
        # shift vel to left label 
        svll = self.cfg.dataset.base.shift_vel_left  # +1 for indexing, -1 for indexes ...

        num_welch_pad =  self.cfg.dataset.base.num_welch_pad  
        assert  num_welch_pad >= 0

        akkashi_padd_margin = num_welch_pad 
 
    

        if seg_dict['data_source'] in ['nano', 'nanoRSC'] :

            seg_start_inwelch = seg_dict['seg_start_inwelch'] 
            seg_end_inwelch = seg_dict['seg_end_inwelch'] 

            if seg_start_inwelch < 0 : 
                print("seg_start_inwelch is negative lol")
                print(seg_dict["seg_type"])
                print(seg_dict["run_id"])
                exit()
                


        # CLIP SAMPLING
        # Test mode :

        if self.clip_len is None :
            # full segment            
            clip_start = seg_start_inwelch + akkashi_padd_margin + max(-svll, 0)  
            clip_end = seg_end_inwelch - max(svll, 0)  -1

        else : 
            # # rdm clips in test

            if seg_dict['contact_mode'] == "slip" :


                # for slip evaluation on short clips, select random clip containing the start of slip 
                ssain, _ = slip_labels_inwelch[0]
                ssain -= svll # add margins
                
                if seg_dict['seg_type'] == "split_slip" :  
                    clip_start = ssain - random.randint(10 , 50 )
                else :  
                    clip_start = ssain - random.randint(50 , 120)
                
                clip_end = clip_start + self.clip_len -1


            elif seg_dict['contact_mode'] == "noslip" : 

                if seg_dict['seg_type'] == 'noslip_record':
                    #  case disabled
                    raise RuntimeError("Legacy branch disabled: seg_type=='noslip_record'")
                
                    # clip_start = random.randint( seg_start_inwelch + akkashi_padd_margin + margin_clip, seg_end_inwelch - self.clip_len )
                    # # clip_start = seg_start_inwelch + 6  # BALAN
                    # clip_end = clip_start + self.clip_len -1


                
                elif 'noslip_run' in seg_dict['seg_type']:
                    # test noslip
                    
                    # event_idx = random.choice(seg_dict['noise_event_index_list'])
                    event_indices = self.sort_event_indices_with_highest_intensity( seg_dict, welch) # select clips with perturbations
                    event_idx_start, event_idx_end = random.choice(event_indices[0:2])


                    
                    clip_start = random.randint( event_idx_start - self.clip_len //4  , event_idx_start)  
                
                    clip_end = clip_start + self.clip_len -1




        seg_dict['clip_start_inwelch'] = clip_start
        seg_dict['clip_end_inwelch'] = clip_end




        data = welch[self.minfreq:self.maxfreq, clip_start  : clip_end+1   ]  
 
  
        dense_torques = seg_dict['dense_torques'][:, clip_start  + self.synchro_offset_torques  : clip_end+1   + self.synchro_offset_torques] # C, L
 



        csa_sv = clip_start + svll #  offset on bench ground  truth to synchronize  signals
        cse_sv = clip_end + svll
        
        dense_slip_label_raw = dense_slip_label
        dense_slip_label = dense_slip_label[csa_sv  : cse_sv+1 ] 
        
        
        dense_noise_label = dense_noise_label[clip_start  : clip_end+1 ] 



        # check clip length, clip_len is the target in loader cfg
        if self.clip_len is not None :
            assert data.shape[1] == self.clip_len #, data.shape
            assert len(dense_slip_label) == self.clip_len #, dense_slip_label.shape




        # preprocessing spectrogram to db
        data = 10*np.log10(data)
         
        self.init_norm_stats()

        # normalization on full_data , feature-level
        mean = self.norm_stats["mean"][self.minfreq:self.maxfreq]
        std = self.norm_stats["std"][self.minfreq:self.maxfreq]

        mean = np.expand_dims(mean, axis=1)
        std = np.expand_dims(std, axis=1)
        data = (data - mean ) / std

        

        sensor_id_embed =  self.sensor_id_embedding (seg_dict)




        assert  dense_slip_label.shape[0] == data.shape[1]




        # SUPPLEMENT INFO, for debug and metrics

        additional_info  = {}

        additional_info['slip_run_id'] = seg_dict['slip_run_id'] 
        additional_info["clip_wlech_limits"] = (seg_dict['clip_start_inwelch'] ,  seg_dict['clip_end_inwelch'] ) 
        additional_info['seg_type'] = seg_dict['seg_type'] 
        additional_info['pze_col_name'] = seg_dict['pze_col_name']
        additional_info["tokyo_fft"] = seg_dict["tokyo_fft"] # for visualization data dump...   TODO remove

        return data,  dense_slip_label, dense_noise_label, dense_torques,  sensor_id_embed, additional_info

         

    def init_norm_stats(self):

        self.norm_stats = {}
        # proper nanoslip
        self.norm_stats["mean"] = np.array([-28.992, -30.353, -36.227, -34.357, -31.735, -34.88 , -38.652,
            -39.062, -39.382, -39.76 , -40.1  , -40.398, -40.486, -40.651,
            -40.997, -41.183, -41.241, -41.385, -41.607, -41.697, -41.677,
            -41.795, -41.793, -41.812, -42.024, -41.966, -41.928, -41.929,
            -41.825, -41.853, -41.77 , -41.778, -41.806, -41.683, -41.629,
            -41.655, -41.623, -41.519, -41.443, -41.521, -41.599, -41.513,
            -41.51 , -41.529, -41.589, -41.694, -41.693, -41.726, -41.659,
            -41.754, -41.711, -41.782, -41.839, -41.863, -41.933, -42.091,
            -42.073, -42.144, -42.223, -42.176, -42.124, -42.209, -42.387,
            -42.429, -42.38 , -42.374, -42.33 , -42.323, -42.064, -42.055,
            -41.998, -42.003, -41.952, -41.944, -41.83 , -41.777, -41.819,
            -41.752, -41.778, -41.613, -41.607, -41.724, -41.593, -41.516,
            -41.475, -41.459, -41.466, -41.425, -41.482, -41.459, -41.515,
            -41.565, -41.644, -41.675, -41.679, -41.726, -41.75 , -41.797,
            -41.89 , -41.985, -41.971, -42.07 , -42.111, -42.211, -42.243,
            -42.204, -42.163, -42.222, -42.268, -42.216, -42.113, -42.202,
            -42.179, -42.054, -42.053, -42.   , -41.983, -41.888, -41.854,
            -41.866, -41.819, -41.722, -41.7  , -41.623, -41.707, -41.659,
            -41.656, -41.68 , -44.577])

        self.norm_stats["std"] = np.array([15.322, 12.28 ,  9.531,  8.066,  7.933,  7.91 ,  8.182,  8.105,
                8.109,  8.055,  7.627,  7.383,  7.291,  7.1  ,  6.985,  6.833,
                6.74 ,  6.639,  6.614,  6.505,  6.408,  6.383,  6.31 ,  6.325,
                6.327,  6.145,  6.086,  6.125,  6.06 ,  6.171,  6.163,  6.331,
                6.186,  6.199,  6.173,  6.071,  6.166,  6.155,  6.179,  6.2  ,
                6.143,  6.147,  6.064,  6.088,  6.087,  6.065,  6.027,  6.094,
                5.973,  6.023,  5.937,  5.98 ,  5.908,  5.928,  5.915,  5.912,
                5.834,  5.845,  5.874,  5.847,  5.79 ,  5.851,  5.928,  5.873,
                5.812,  5.804,  5.786,  5.879,  5.741,  5.826,  5.77 ,  5.866,
                5.882,  5.871,  5.899,  5.906,  5.958,  5.942,  6.   ,  5.85 ,
                5.911,  6.051,  5.999,  5.988,  5.936,  5.972,  5.936,  5.979,
                5.978,  5.938,  5.985,  5.95 ,  5.992,  5.933,  5.881,  5.892,
                5.859,  5.831,  5.825,  5.871,  5.877,  5.805,  5.807,  5.839,
                5.748,  5.805,  5.759,  5.797,  5.864,  5.814,  5.778,  5.803,
                5.839,  5.859,  5.823,  5.859,  5.91 ,  5.954,  5.926,  5.92 ,
                6.   ,  5.972,  5.996,  5.94 ,  6.074,  6.049,  6.067,  5.987,
                9.899])
        



    def  sensor_id_embedding(self, seg_dict):

    
        pze_col_name_fnum = seg_dict['pze_col_name'].split('.')[-2][-1]
        pze_col_name_phx = seg_dict['pze_col_name'].split('.')[-1][-1]
        pze_col_name_embedding = [int(pze_col_name_fnum), int(pze_col_name_phx)]
 

        pze_col_name_embedding_onehot = [0] * 12

        if pze_col_name_embedding == [0, 0] : 
            pze_col_name_embedding_onehot[0] = 1
        elif pze_col_name_embedding == [0, 1] : 
            pze_col_name_embedding_onehot[1] = 1
        elif pze_col_name_embedding == [0, 2] : 
            pze_col_name_embedding_onehot[2] = 1


        elif pze_col_name_embedding == [1, 0] : 
            pze_col_name_embedding_onehot[3] = 1
        elif pze_col_name_embedding == [1, 1] : 
            pze_col_name_embedding_onehot[4] = 1
        elif pze_col_name_embedding == [1, 2] : 
            pze_col_name_embedding_onehot[5] = 1


        elif pze_col_name_embedding == [2, 0] : 
            pze_col_name_embedding_onehot[6] = 1
        elif pze_col_name_embedding == [2, 1] : 
            pze_col_name_embedding_onehot[7] = 1
        elif pze_col_name_embedding == [2, 2] : 
            pze_col_name_embedding_onehot[8] = 1


        elif pze_col_name_embedding == [3, 0] : 
            pze_col_name_embedding_onehot[9] = 1
        elif pze_col_name_embedding == [3, 1] : 
            pze_col_name_embedding_onehot[10] = 1
        elif pze_col_name_embedding == [3, 2] : 
            pze_col_name_embedding_onehot[11] = 1

        return pze_col_name_embedding_onehot




    def sort_event_indices_with_highest_intensity(self,  seg_dict, data):


        # window_size = 5
        data = data[1:17]  # Low freqs only

        event_intensities = {}

        for start_idx, end_idx in self.zip_noise_events_indexes(seg_dict) :
            # Define the window around the event index

            # Extract the window data
            window_data = data[2:, start_idx:end_idx] # remove  freq  bin 0  (DC), before data croping
            
            # Flatten the window data to 1D and sort by intensity
            sorted_intensities = np.sort(window_data.flatten())[::-1]  # Sort in descending order
            
            # Calculate the mean of the top 10 pixel intensities
            top_10_mean = np.mean(sorted_intensities[:10])
            
            # Store the event index along with its mean intensity
            event_intensities[(start_idx, end_idx)] = top_10_mean
        
        # Sort the event indices by mean intensity in descending order
        sorted_event_indices = sorted(event_intensities.keys(), key=lambda k: event_intensities[k], reverse=True)
            
        return sorted_event_indices





    def zip_noise_events_indexes(self, seg_dict):
        
        # for noslip_run only
        # dense labels from event idxes

        if seg_dict['seg_type'] == "noslip_run_airmotion" : 
            # about 2s
            start_indices = seg_dict['airmotion_traj_start_index_list']
            stop_indices = seg_dict['airmotion_traj_end_index_list']
            start_indices = [si+5 for si in start_indices]


        if seg_dict['seg_type'] == "noslip_run_torques" : 
            # about 0.5s, ambiguous
            start_indices = seg_dict['noise_event_index_list']
            start_indices = [si+2 for si in start_indices]        
            stop_indices = [si+12 for si in start_indices]

        noise_events_start_stop_indices = zip(start_indices, stop_indices)
        return noise_events_start_stop_indices


    def build_dense_noise_label(self, seg_dict, noise_events_start_stop_indices):

        
        array_length = seg_dict["tokyo_fft"].shape[1]

        # Initialize the array with zeros
        binary_array = np.zeros(array_length, dtype=int)

        # Set the ranges between start and stop indices to 1
        for start, stop in noise_events_start_stop_indices:
            binary_array[start:stop] = 1

        assert np.any(binary_array)

        return binary_array
    
