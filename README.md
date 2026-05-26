Real-Time Driver Fatigue Monitoring SystemAn end-to-end multi-task deep learning framework designed to detect driver drowsiness and mitigate road accidents. The system processes facial image streams to simultaneously track two critical biometrics—Eye State (Open vs. Closed) and Mouth State (Yawn vs. No_Yawn)—and fuses them into a continuous 3-Level Fatigue Progression Curve for proactive safety alerts.📌 Project Architecture & PipelineThe pipeline transitions raw, unstructured visual inputs into actionable, sequential driver safety metrics through the following phases:  [ Raw Image Data ]
          │
          ▼
┌──────────────────────────┐
│ Data Augmentation Layer  │ ──► (Horizontal Flips, 10° Rotations, Brightness Shifts)
└─────────┬────────────────┘
          │
          ▼
┌──────────────────────────┐
│  MobileNetV2 Backbone    │ ──► (Top 60% Unfrozen for Fine-Tuning Facial Geometry)
└─────────┬────────────────┘
          │
          ▼
┌──────────────────────────┐     ┌──────────────────────────┐
│   Eye Monitoring Head    │     │   Mouth Monitoring Head  │
│  (Closed / Open State)   │     │ (Yawn / No_Yawn State)   │
└─────────┬────────────────┘     └─────────┬────────────────┘
          │                                │
          └───────────────┬────────────────┘
                          │ (Softmax Probabilities)
                          ▼
             ┌──────────────────────────┐
             │   Decision Fusion Logic  │
             └────────────┬─────────────┘
                          │
                          ▼
            [ 3-Stage Progression Timeline ]
              (Alert ──► Mild ──► Severe)
Preprocessing & GPU Augmentation: Standardizes inputs to $224 \times 224 \times 3$. Includes dynamic horizontal flips, up to $10^\circ$ rotations, and brightness modifications to simulate night driving and sun glare.Dual-Head Feature Extraction: Leverages a pre-trained MobileNetV2 backbone with its top 60% of layers unfrozen to capture subtle facial transformations (e.g., eyelid narrowing and lip contours).Decision Fusion Engine: Maps parallel classification probabilities to structured fatigue levels rather than isolated frame states.⚙️ Decision Fusion Logic (3-Stage Mapping)Individual frame predictions are translated into real-time hazard levels using a rule-based inference structure:Level 0 (Alert): Triggered when the system detects the eye is Open AND the mouth shows No_Yawn.Level 1 (Mild Fatigue): Triggered when the eyes are open but an active Yawn is identified.Level 2 (Severe Fatigue): Critical safety override. Triggered immediately whenever eyes are detected as Closed, regardless of mouth status.📊 Evaluation & Comparative MetricsThe system was evaluated on an unseen testing set consisting of 439 frames to benchmark a Custom CNN (built from scratch) against the optimized Transfer Learning model.Global ComparisonCustom CNN Perfect Frame Accuracy: 26.88% (Suffered from catastrophic class collapse on mouth states)MobileNetV2 Perfect Frame Accuracy: 60.36% (Achieved a 125% improvement in overall system reliability)MobileNetV2 Detailed Classification PerformanceClassification Task / HeadClass LabelPrecisionRecallF1-ScoreSupportTASK 1: EYE STATEClosed1.000.700.82329Open0.530.990.69110TASK 2: MOUTH STATENo_Yawn0.990.780.87330Yawn0.600.970.74109🛠️ Optimization Strategy SummaryTo eliminate majority-class guessing and stabilize training on small datasets, five specific optimizations were integrated:Dynamic Spatial Augmentation: Embedded directly into the model graph to prevent network memorization.Layer Unfreezing Strategy: Unfroze the top 60% of MobileNetV2 base layers to adapt pre-trained weights to human facial details.Low Learning Rate Fine-Tuning: Applied an Adam optimizer at a controlled speed of $1 \times 10^{-5}$ to adjust weights without destroying core edge filters.Task-Priority Weighting: Biased loss weights towards eye state tracking ($1.5 \times$ multiplier) to prioritize safety-critical triggers.Pristine Generator Alignment: Re-engineered the testing data pipeline to extract un-shuffled ground truth, avoiding artificial random noise metrics.🚀 Getting StartedPrerequisitesBashpip install tensorflow numpy matplotlib seaborn scikit-learn split-folders
UsageDataset Structuring: Place your raw images into class folders inside an input directory and run the data splitting pipeline block:Pythonimport splitfolders
splitfolders.ratio('path/to/Input_Folder', output='path/to/Output_Folder', seed=42, ratio=(.80, .10, .10))
Training & Evaluation: Execute the master notebook code to run the data augmentation steps, fit the fine-tuned model at a learning rate of 1e-5, and print the complete classification metrics report.⚠️ Limitations & Future OutlookSquinting Noise: Intense smiling or sharp oncoming headlight glare can briefly lower vertical eyelid distances, causing minor false positives for the Closed state.Extreme Head Tilts: Lateral rotations exceeding 45° can obscure landmarks on the far side of the face.Next Steps: Introduce temporal models such as Long Short-Term Memory (LSTM) or GRU networks to track feature modifications across consecutive frame windows rather than evaluating individual moments in isolation.
