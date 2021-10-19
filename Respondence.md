### Concern #1: The paper needs to be completely rewritten from scratch. Explain the objectives, the techniques adopted, and the evaluation.
- Authors’ response:

Objectives: 

While SORT and DeepSORT are not optimized for both accuracy and processing time, ReSORT is developed with three main purposes: reducing the identity switching, recovering the camera-disappeared identities, and speeding up the processing time. The first two purposes are to improve the robustness and reduce the number of alarms in the face recognition system, while the last purpose is to guarantee real-time performance. 

- ReSORT method:  

ReSORT takes the detection results of a frame, i.e. bounding box (face) information, and outputs the identity (ID) for each bounding box. The overview of ReSORT is shown in Figure 1. ReSORT first uses the Kalman filter to predict the current locations of the faces detected in the previous frame. It then associates the predicted bounding box with the detected bounding boxes in the current frame using IoU matching. A similarity matching (SM) block is proposed to append after the IoU matching block with the aims to: 

- Increase the object track association reliability. 

- Retrieve the previously appeared ID to associate or register new faces. 

- Reduce IDs switch counts. 

Precisely, the SM block receives faces, which are not determined the ID by the IoU matching block, and returns the tracking identities for those faces. It first uses the Face-pose block to estimate the orientation of the face being half or frontal by using five facial landmarks, see Eq. (1). The frontal faces are passed to the recovery block, which stores the embedded vectors of the previously appeared IDs. The embedding vectors of the fontal faces are extracted via a face extractor. If the embedded face vector is similar enough to a particular stored face ID, it is assigned to that ID. Otherwise, a new ID is created. Furthermore, the life cycle of a tracker in ReSORT, which is based on one state variable with four values: tentative, unconfirmed, confirmed, and deleted, is managed to avoid missing the slanting faces and ensure the ID-recovery ability. 

### Concern #2:  Compare it quantitatively with other work.
- Authors’ response:

We used CLEAR MOT metrics to evaluate the tracking methods on three public datasets with various configurations of frame rate and resolution. We also introduce three new metrics to measure the correctness of face re-identification, including IDnew, TReID, TReRate. Tables I, II, and III shows the quantative results using the CLEAR MOT metrics and three introduced metrics. The experimental results demonstrate that ReSORT reduces significantly ID-switching counts, while enabling real-time performance.  

### Concern #3: The paper is insufficiently clear to allow reproduction although the code will be published.
- Authors’ response:

Please see the README in this repository to reproduce the proposed ReSORT method.