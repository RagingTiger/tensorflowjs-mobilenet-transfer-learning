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

const video = document.getElementById('webcam');
const enableCamButton = document.getElementById('enableCam');
const class1DataButton = document.getElementById('class1Data');
const class2DataButton = document.getElementById('class2Data');
const trainButton = document.getElementById('train');


enableCamButton.addEventListener('click', enableCam);
class1DataButton.addEventListener('click', gatherDataClass1);
class2DataButton.addEventListener('click', gatherDataClass2);
trainButton.addEventListener('click', trainAndPredict);


// Check if webcam access is supported.
function hasGetUserMedia() {
  return !!(navigator.mediaDevices &&
    navigator.mediaDevices.getUserMedia);
}


// Enable the live webcam view and start classification.
function enableCam() {
  if (hasGetUserMedia()) {
    
    
  } else {
    console.warn('getUserMedia() is not supported by your browser');
  }
  
  // getUsermedia parameters.
  const constraints = {
    video: true
  };

  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
    video.srcObject = stream;
    video.addEventListener('loadeddata', predictWebcam);
  });
}