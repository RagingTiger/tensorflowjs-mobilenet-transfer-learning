TensorFlow.js Boilerplate
=================

This very simple skeleton simply loads in TensorFlow.js and prints out the version once loaded to the DOM.

From these humble beginnings you can do some really great things. 

Feel free to fork this and use it as a quick way to start coding with TensorFlow.js directly or following some other tutorial that needs TensorFlow.js to run.


Your Project
------------

### ← index.html

We simply have a script tag in our HTML to grab the latest version of TensorFlow.js

In this case we simply reference the following:

```HTML
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js" type="text/javascript"></script>
```

However, if you want to pull in a particular version of TensorFlow.js you can do so like this:

```HTML
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.4.0/dist/tf.min.js" type="text/javascript"></script>
```

### ← script.js

This simply grabs a reference to a paragraph in the DOM and then prints out the TensorFlow.js version number to it once loaded.


-------------------
