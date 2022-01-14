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

ENABLE_CAM_BUTTON.addEventListener('click', enableCam);
CLASS_1_DATA_BUTTON.addEventListener('mousedown', gatherDataClass1);
CLASS_1_DATA_BUTTON.addEventListener('mouseup', gatherDataClass1);

CLASS_2_DATA_BUTTON.addEventListener('mousedown', gatherDataClass2);
CLASS_2_DATA_BUTTON.addEventListener('mouseup', gatherDataClass2);
TRAIN_BUTTON.addEventListener('click', trainAndPredict);

let model = undefined;
let mobilenet = undefined;
let gather_data_state = 0;
let videoPlaying = false;


async function loadMobileNetFeatureModel() {
  mobilenet = await tf.loadGraphModel('https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1', {fromTFHub: true});
  STATUS.innerText = 'MobileNet v3 loaded successfully!';
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
      video: true
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
      VIDEO.srcObject = stream;
      video.addEventListener('loadeddata', function() {
        videoPlaying = true;
      });
    });
  } else {
    console.warn('getUserMedia() is not supported by your browser');
  }
}


function dataGatherLoop() {
  // Only gather data if webcam is on and working.
  if (videoPlaying) {
    // Ensure tensors are cleaned up
    tf.tidy(function() {
      let videoFrameAsTensor = tf.browser.fromPixels(VIDEO);
      videoFrame.print();
    });

    window.requestAnimationFrame(dataGatherLoop);
  }
}


function gatherDataClass1() {
  gather_data_state = (gather_data_state === 0) ? 1 : 0;
  
  if(gather_data_state !== 0) {
    dataGatherLoop();
  }
}


function gatherDataClass2() {
  gather_data_state = (gather_data_state === 0) ? 2 : 0;
  console.log(gather_data_state);
}


function trainAndPredict() {
  
}