import numpy as np
import cv2
import pickle
import time
import torch
from utils import desc_map, desc_sampling, error_calculator


def train_one_epoch(model_bbone: torch.nn.Module,optimizer: torch.optim.Optimizer, cr_check, data_dir, img_files, cal_db, vis_pairs_mod, samp_pts, threshold, epoch, output_dir, ransac_type):
    temp = 0
    lamda1 = 1
    mean_loss_heap = []
    min_loss_heap = []
    nfeatures = 2000
    # loweRatio = args.ratio

    counter = 0
    start = time.time()
    file_saver = True
    for i, vis_pair in enumerate(vis_pairs_mod):
        #     for ite in range(batch_size):
        counter += 1
        img_stack = []

        img1_idx = vis_pair[0]
        img2_idx = vis_pair[1]
        db_index = vis_pair[2]

        img1 = cv2.imread(data_dir[db_index] + img_files[db_index][img1_idx][:-1])
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype('float32') / 255.
        img_stack.append(img1)

        img2 = cv2.imread(data_dir[db_index] + img_files[db_index][img2_idx][:-1])
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype('float32') / 255.
        img_stack.append(img2)

        # print("Debug: Image shape: ", img_arr.shape, flush=True)
        # Debug purpose visualization starts
        # import matplotlib.pyplot as plt
        # from matplotlib.patches import ConnectionPatch
        # f, axarr = plt.subplots(1,2, figsize=(20,10))
        # # print("Image name, size, shape: ", frame_ids[b], sample['imgs'][0][b].shape, sample['image_size1'][b], flush=True)
        # axarr[0].imshow(img1, cmap='gray')
        # axarr[1].imshow(img2, cmap='gray')
        # # Set title
        # plt.suptitle(f"sized {img1.shape} and {img2.shape}")
        # img_name1 = img_files[db_index][img1_idx][:-1].split('/')[-1]
        # img_name2 = img_files[db_index][img2_idx][:-1].split('/')[-1]
        # plt.savefig(f"/cluster/project/cvg/students/shikim/fully-differentiable-global-sfm/visualizations_rfp/{img_name1}+{img_name2}.png")
        # plt.clf()
        # Debug purpose visualization ends

        img_arr = np.asarray(img_stack)

        grad_stack = []
        desc_grad_stack = []
        loss_stack = []

        heatmap, coarse_desc, log_map, coarse_org = model_bbone.run(img_arr)
        for itera in range(3):
            pts_stack_1, desc_stack_1, inv_prob_arr = model_bbone.key_pt_sampling(img_arr, heatmap, samp_pts,
                                                                                  coarse_desc)
            matched, desc_shape = desc_map(pts_stack_1, desc_stack_1, cr_check)
            desc_grad_mini = []
            loss_mini = []
            for desc_itera in range(3):
                pts_stack, desc_stack, desc_grad, match_samples = desc_sampling(matched, pts_stack_1, desc_stack_1)
                # print("Image 1 and 2 path: ", data_dir[db_index] + img_files[db_index][img1_idx][:-1],
                #       data_dir[db_index] + img_files[db_index][img2_idx][:-1], flush=True)
                # print("Image 1 and Image 2 Shapes: ", img1.shape, img2.shape, flush=True)
                loss = error_calculator(pts_stack, desc_stack, lamda1, match_samples, cal_db, img1_idx, img2_idx,
                                         db_index, threshold, img1.shape, img2.shape, ransac_type)
                if (np.isnan(loss[0, 0]) == True):
                    loss[0, 0] = temp
                loss_stack.append(loss[0, 0])
                #         print(np.sum(loss))
                temp = loss[0, 0]
                loss_mini.append(temp)
                desc_grad_mini.append(desc_grad)
                grad_stack.append(inv_prob_arr)

            loss_arr_mini = np.asarray(loss_mini)
            desc_grad_arr_mini = np.asarray(desc_grad_mini)
            mean_loss_mini = np.mean(loss_arr_mini)
            loss_arr_mini = loss_arr_mini - mean_loss_mini
            desc_grad_upd = np.sum(loss_arr_mini[:, np.newaxis, np.newaxis, np.newaxis] * desc_grad_arr_mini,
                                   axis=0)
            desc_grad_stack.append(
                torch.autograd.grad(desc_stack_1, coarse_desc, torch.from_numpy(desc_grad_upd / 3.0).float())[
                    0].cpu().numpy())

        loss_arr = np.asarray(loss_stack)
        grad_arr = np.asarray(grad_stack)
        desc_grad_arr = np.asarray(desc_grad_stack)
        mean_loss = np.mean(loss_arr)
        std_loss = np.std(loss_arr)
        mean_loss_heap.append(mean_loss)
        min_loss_heap.append(np.amin(loss_arr))
        print(mean_loss, np.amin(loss_arr), std_loss)
        loss_arr = loss_arr - mean_loss

        update = np.sum(loss_arr[:, np.newaxis, np.newaxis, np.newaxis] * grad_arr, axis=0)
        #     update_desc = np.sum(loss_arr[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]*desc_grad_arr, axis = 0)
        update_desc = np.sum(desc_grad_arr, axis=0)
        #     print("iteration : ", ite)
        print("epoch : ", epoch, "iteration :", i)
        #     print(len(loss_arr), len(loss_mini), len(grad_stack), len(desc_grad_stack))

        torch.autograd.backward([log_map, coarse_org], [torch.from_numpy(update / 9.0).cuda().float(),
                                                        torch.from_numpy(
                                                            (lamda1 * update_desc) / 3.0).cuda().float()])

        optimizer.step()
        #        scheduler.step()

        optimizer.zero_grad()

    end = time.time()
    print("time taken for this epoch: ", end - start)
    if epoch % 10 == 0:
        with open(output_dir + "/mean_loss_heap_ransac.txt", "wb") as fp:  # Pickling
            pickle.dump(mean_loss_heap, fp)
        with open(output_dir + "/min_loss_heap_ransac.txt", "wb") as fp:  # Pickling
            pickle.dump(min_loss_heap, fp)
        torch.save(model_bbone.net.state_dict(), output_dir + '/ransac_cross_check_True.pth')

