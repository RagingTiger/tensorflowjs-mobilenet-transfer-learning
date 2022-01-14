/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const STATUS = document.getElementById('status');
const VIDEO = document.getElementById('webcam');
const ENABLE_CAM_BUTTON = document.getElementById('enableCam');
const CLASS_1_DATA_BUTTON = document.getElementById('class1Data');
const CLASS_2_DATA_BUTTON = document.getElementById('class2Data');
const TRAIN_BUTTON = document.getElementById('train');
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;

ENABLE_CAM_BUTTON.addEventListener('click', enableCam);
CLASS_1_DATA_BUTTON.addEventListener('mousedown', gatherDataClass1);
CLASS_1_DATA_BUTTON.addEventListener('mouseup', gatherDataClass1);

CLASS_2_DATA_BUTTON.addEventListener('mousedown', gatherDataClass2);
CLASS_2_DATA_BUTTON.addEventListener('mouseup', gatherDataClass2);
TRAIN_BUTTON.addEventListener('click', trainAndPredict);

const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [1024], units: 128, activation: 'relu'}));
model.add(tf.layers.dense({units: 2, activation: 'softmax'}));
// Compile the model with the defined optimizer and specify a loss function to use.
model.compile({
  optimizer: 'adam', // Adam changes the learning rate over time which is useful.
  loss: 'categoricalCrossentropy', // As this demo supports just 2 classes, this is a binary classification problem.
  metrics: ['accuracy']  // As this is a classifcation problem you can ask to record accuracy in the logs too!
});

let mobilenet = undefined;
let gatherDataState = 0;
let videoPlaying = false;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];


async function loadMobileNetFeatureModel() {
  mobilenet = await tf.loadGraphModel('https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1', {fromTFHub: true});
  STATUS.innerText = 'MobileNet v3 loaded successfully!';
  // Warm up the model by passing zeros through it once.
  tf.tidy(function () {
    mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
  });
}

loadMobileNetFeatureModel();


// Check if webcam access is supported.
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}


// Enable the live webcam view and start classification.
function enableCam() {
  if (hasGetUserMedia()) {
    // getUsermedia parameters.
    const constraints = {
      video: true,
      width: 640, 
      height: 480 
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
      VIDEO.srcObject = stream;
      VIDEO.addEventListener('loadeddata', function() {
        videoPlaying = true;
        ENABLE_CAM_BUTTON.classList.add('removed');
      });
    });
  } else {
    console.warn('getUserMedia() is not supported by your browser');
  }
}


function dataGatherLoop() {
  // Only gather data if webcam is on and working.
  if (videoPlaying && gatherDataState !== 0) {
    // Ensure tensors are cleaned up.
    let imageFeatures = tf.tidy(function() {
      // Grab pixels from current VIDEO frame, and then divide by 255 to normalize.
      let videoFrameAsTensor = tf.browser.fromPixels(VIDEO).div(255);
      // Resize video frame tensor to be 224 x 224 pixels which is needed by MobileNet for input.
      let resizedTensorFrame = tf.image.resizeBilinear(
          videoFrameAsTensor, 
          [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
          true
      );
      return mobilenet.predict(resizedTensorFrame.expandDims()).squeeze().arraySync();
    });
    
    trainingDataInputs.push(imageFeatures);
    trainingDataOutputs.push(gatherDataState);
    
    // Intialize array index element if currently undefined.
    if (examplesCount[gatherDataState] === undefined) {
      examplesCount[gatherDataState] = 0;
    }
    
    // Increment counts of examples for user interface to show.
    examplesCount[gatherDataState]++;
    STATUS.innerText = 'Class 1 Data count: ' + examplesCount[1] + ', Class 2 Data count: ' + examplesCount[2];

    window.requestAnimationFrame(dataGatherLoop);
  }
}


function gatherDataClass1() {
  gatherDataState = (gatherDataState === 0) ? 1 : 0;
  dataGatherLoop();
}


function gatherDataClass2() {
  gatherDataState = (gatherDataState === 0) ? 2 : 0;
  dataGatherLoop();
}


async function trainAndPredict() {
  let oneHotOutputs = tf.oneHot(tf.tensor1d(trainingDataOutputs, 'int32'), 2);
  let inputsAsTensors = tf.tensor2d(trainingDataInputs);
  
  let results = await model.fit(inputsAsTensors, oneHotOutputs, {
    shuffle: true,
    batchSize: 10,
    epochs: 50,
    callbacks: {onEpochEnd: logProgress}
  });

  predict();
}

function predict() {
  tf.tidy(function() {
    // Grab pixels from current VIDEO frame, and then divide by 255 to normalize.
    let videoFrameAsTensor = tf.browser.fromPixels(VIDEO).div(255);
    // Resize video frame tensor to be 224 x 224 pixels which is needed by MobileNet for input.
    let resizedTensorFrame = tf.image.resizeBilinear(
        videoFrameAsTensor, 
        [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
        true
    );

    let imageFeatures = mobilenet.predict(resizedTensorFrame.expandDims());
    let prediction = model.predict(imageFeatures);
  });
}

function logProgress(epoch, logs) {
  console.log('Data for epoch ' + epoch, logs);
}
