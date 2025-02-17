const spawn = require("child_process").spawn;
const Speaker = require('speaker');
var tf = require("@tensorflow/tfjs-node-gpu")
// prepare 2 child processes
const recordProcess = spawn("arecord", ["-r", "4000"]);
const speaker = new Speaker({
  channels: 1,          // 2 channels
  bitDepth: 8,         // 16-bit samples
  sampleRate: 4000,     // 44,100 Hz sample rate
});

var model = tf.sequential();

model.add(tf.layers.dense({ units: 1024*12, inputShape: [1], activation: "relu" }));

model.add(tf.layers.dense({ units: 1024*12, activation: "relu" }));
model.add(tf.layers.dense({ units: 1024*12, activation: "relu" }));


model.add(tf.layers.dense({ units: 1, activation: "relu" }));
var load = false;
(async () => {
if (load)
        try {
	model = await tf.loadLayersModel('file://my-model/model.json');
	} catch(e) {
		console.log(e)
	}

model.compile({
  loss: tf.losses.meanSquaredError,
  optimizer: tf.train.sgd(0.00001),
  metrics: ['MAE']
});
// get debug info if you want
var  canTrain = true
var data
var lastData
var pad_array = function(arr,len,fill) {
  return arr.concat(Array(len).fill(fill)).slice(0,len);
}


recordProcess.stdout.on('data', function (d) {

   data = d
   if (!lastData) lastData = data

   if (lastData.length > data.length) {
        data = pad_array(Array.from(data), lastData.length, 0)
   } else {
        lastData = pad_array(Array.from(lastData), data.length, 0)
   }
   var a = tf.tensor1d(data);
   var b = tf.tensor1d(lastData);
   if (canTrain) {
   
   model.fit(b, a).then(async () => { canTrain = true; await model.save('file://my-model'); }).catch(e => console.log )

   canTrain = false
   }
   lastData = data
   
   var p = model.predict(a)
   p = p.dataSync()
   buf = new Uint8Array(p.length)
   for (var i = 0; i < p.length; i++) {
   	buf[i] = p[i] * 5
   }
   
   
   speaker.write(buf)
});
/*
recordProcess.stderr.on('data', function (data) {
  console.log('Error: ' + data);
});
recordProcess.on('close', function (code) {
  console.log('arecord closed: ' + code);
});
*/

// this seems like a good idea, but might not be needed
process.on("exit", (code) => {
  console.log(`About to exit with code: ${code}`);
  recordProcess.kill("SIGTERM");
});
setInterval(async () => {
	model.save("file://my-model2")
}, 1000*60)

})()
