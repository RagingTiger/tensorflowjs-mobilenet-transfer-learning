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
const RESET_BUTTON = document.getElementById('reset');
const TRAIN_BUTTON = document.getElementById('train');
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;

ENABLE_CAM_BUTTON.addEventListener('click', enableCam);

let dataCollectorButtons = document.querySelectorAll('button.dataCollector');
for (let i = 0; i < dataCollectorButtons.length; i++) {
  dataCollectorButtons[i].addEventListener('mousedown', gatherDataForClass);
  dataCollectorButtons[i].addEventListener('mouseup', gatherDataForClass);
}

TRAIN_BUTTON.addEventListener('click', trainAndPredict);
RESET_BUTTON.addEventListener('click', reset);

let model = tf.sequential();
model.add(tf.layers.dense({inputShape: [1024], units: 128, activation: 'relu'}));
model.add(tf.layers.dense({units: 2, activation: 'softmax'}));

model.summary();

// Compile the model with the defined optimizer and specify a loss function to use.
model.compile({
  optimizer: 'adam', // Adam changes the learning rate over time which is useful.
  loss: 'binaryCrossentropy', // As this demo supports just 2 classes, this is a binary classification problem.
  metrics: ['accuracy']  // As this is a classifcation problem you can ask to record accuracy in the logs too!
});

let mobilenet = undefined;
let gatherDataState = -1;
let videoPlaying = false;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false;


/**
 * Loads the MobileNet model and warms it up so ready for use.
 **/
async function loadMobileNetFeatureModel() {
  mobilenet = await tf.loadGraphModel('https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1', {fromTFHub: true});
  STATUS.innerText = 'MobileNet v3 loaded successfully!';
  // Warm up the model by passing zeros through it once.
  tf.tidy(function () {
    mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
  });
}

loadMobileNetFeatureModel();


/**
 * Check if getUserMedia is supported for webcam access.
 **/
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}


/**
 * Enable the webcam with video constraints applied.
 **/
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


/**
 * When a button used to gather data is pressed, record feature vectors along with class type to arrays.
 **/
function dataGatherLoop() {
  // Only gather data if webcam is on and a relevent button is pressed.
  if (videoPlaying && gatherDataState !== STOP_DATA_GATHER) {
    // Ensure tensors are cleaned up.
    let imageFeatures = tf.tidy(function() {
      // Grab pixels from current VIDEO frame.
      let videoFrameAsTensor = tf.browser.fromPixels(VIDEO);
      console.log(videoFrameAsTensor.shape);
      // Resize video frame tensor to be 224 x 224 pixels which is needed by MobileNet for input.
      let resizedTensorFrame = tf.image.resizeBilinear(
          videoFrameAsTensor, 
          [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
          true
      );
      
      let normalizedTensorFrame = resizedTensorFrame.div(255);
      
      return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
    });

    trainingDataInputs.push(imageFeatures);
    trainingDataOutputs.push(gatherDataState);
    
    // Intialize array index element if currently undefined.
    if (examplesCount[gatherDataState] === undefined) {
      examplesCount[gatherDataState] = 0;
    }
    
    // Increment counts of examples for user interface to show.
    examplesCount[gatherDataState]++;
    STATUS.innerText = 'Class 1 Data count: ' + examplesCount[0] + ', Class 2 Data count: ' + examplesCount[1];

    window.requestAnimationFrame(dataGatherLoop);
  }
}


/**
 * Handle Data Gather for button mouseup/mousedown.
 **/
function gatherDataForClass() {
  let classNumber = parseInt(this.getAttribute('data-class'));
  gatherDataState = (gatherDataState === STOP_DATA_GATHER) ? classNumber : STOP_DATA_GATHER;
  dataGatherLoop();
}


async function trainAndPredict() {
  predict = false;
  tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);

  let oneHotOutputs = tf.oneHot(tf.tensor1d(trainingDataOutputs, 'int32'), 2);

  let inputsAsTensors = tf.stack(trainingDataInputs);
  
  let results = await model.fit(inputsAsTensors, oneHotOutputs, {
    shuffle: true,
    batchSize: 5,
    epochs: 10,
    callbacks: {onEpochEnd: logProgress}
  });
  predict = true;
  predictLoop();
}

function predictLoop() {
  if(predict) {
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
      let prediction = model.predict(imageFeatures).squeeze();
      let highestIndex = prediction.argMax().arraySync();
      let predictionArray = prediction.arraySync();

      STATUS.innerText = 'Prediction: ' + highestIndex + ' with ' + (predictionArray[highestIndex] * 100)+ '% confidence';
    });

    window.requestAnimationFrame(predictLoop);
  }
}

function logProgress(epoch, logs) {
  console.log('Data for epoch ' + epoch, logs);
}


function reset() {
  predict = false;
  examplesCount.splice(0);
  trainingDataInputs.splice(0);
  trainingDataOutputs.splice(0);
  STATUS.innerText = STATUS.innerText = 'Class 1 Data count: 0, Class 2 Data count: 0';
}