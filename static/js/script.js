//import {Chart} from './Chart.js';
//https://cdn.jsdelivr.net/npm/chart.js@2.8.0
const canvas = document.querySelector('#draw');
//const bar = document.querySelector('#chart').getContext('2d');
console.log('here')
canvas.height = 300;
canvas.width = 300;
var invisible = true;
const c = canvas.getContext('2d');
c.strokeStyle = "black";
var mousePressed = false;
var lastX=0;
var lastY=0;
var Xmax = -100000;
var Xmin = 100000;
var Ymax = -100000;
var Ymin = 100000;
canvas.addEventListener('mousedown',function (e) {
    let rect = canvas.getBoundingClientRect();
    var x = e.clientX - rect.left;
    var y = e.clientY - rect.top;
    //e.clientX and e.x are the same
        mousePressed = true;
        Draw(x, y, false);
    });

    
canvas.addEventListener('mousemove',function (e) {

        if (mousePressed) {
            let rect = canvas.getBoundingClientRect();
    var x = e.clientX - rect.left;
    var y = e.clientY - rect.top;
        Draw(x, y, true);
        }
    });

    
canvas.addEventListener('mouseup',function (e) {
        mousePressed = false;
    });
canvas.addEventListener('mouseleave',function (e) {
        mousePressed = false;
    });

/*
function download() {
    var download = document.getElementById("download");
    var image = document.getElementById("myCanvas").toDataURL("image/png")
        .replace("image/png", "image/octet-stream");
    download.setAttribute("href", image);
    //download.setAttribute("download","archive.png");
    }
*/
function Draw(x, y, isDown) {
    if (isDown) {
        c.beginPath();
        c.strokeStyle = "#00000";
        c.lineWidth = 18;
        c.lineCap = "square";
        c.lineJoin = "round";
        
        c.moveTo(lastX, lastY);
        c.lineTo(x, y);
        c.closePath();
        c.stroke();
    }
    lastX = x; lastY = y;
    if (lastX > Xmax){ Xmax = lastX;}
    if (lastX < Xmin){ Xmin = lastX;}
    if (lastY > Ymax){ Ymax = lastY;}
    if (lastY < Ymin){ Ymin = lastY;}
}
	
function clearArea() {
    // Use the identity matrix while clearing the canvas
    //c.setTransform(1, 0, 0, 1, 0, 0);
    Xmax = -100000;
    Xmin = 100000;
    Ymax = -100000;
    Ymin = 100000;
    c.clearRect(0, 0, c.canvas.width, c.canvas.height);
    document.querySelector('#res').textContent = '';
}
const select = document.querySelector('#slct');
let header =  document.querySelector('h1');
let acc =  document.querySelector('#accuracy');
let algo = 'cnn1';

//cnn accuracy : 97.50
//resnet accuracy : 97.86 
//rnn accuracy :  93.96
const accuracy = {
    'cnn1' : 96.88,
    'cnn2' : 95.8,
    'resnet' : 97.86,
    'rnn' : 93.96
}
const names = {
    'cnn1' : 'Convolutional Neural Network (1)',
    'cnn2' : 'Convolutional Neural Network (2)',
    'resnet' :'Residual Neural Network',
    'rnn' : 'Recurret Neural Network'
}
var container = document.querySelector('#container');
var chhart = document.querySelector('#mychart');
var clicked = true;
acc.textContent = 'Test accuracy of the model is : ' + '/';
select.addEventListener('change',(e)=>{
    algo = e.target.value;
    header.textContent = names[e.target.value];
    acc.textContent = 'Test accuracy of the model is : ' + accuracy[algo];
    if (invisible){
        container.style.transition = "opacity 2s";
        container.style.opacity = '1';
        container.style.transition = "margin-top 1s";
        container.style.marginTop = '0px';
        invisible = false;

    }

});
const button = document.querySelector('#btn');
const correct = document.querySelector('#correct');
canvas.style.backgroundColor = "#ffffff";
const letters = ['أ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ',
                'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 
                'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي']
const del = document.querySelector('#delete');
var capture = false;
button.addEventListener('click' , function(){
        
        if (capture) {
        var canvas3 = canvas;
        let w = Xmax - Xmin;
        let h = Ymax - Ymin;
        c.rect(Xmin-20, Ymin-20, w + 40 ,h + 40);
        c.lineWidth = "0.5";
        c.strokeStyle = "black";
        c.stroke();
        var canvas1 = document.createElement("canvas");
        

        canvas1.width =50;
        canvas1.height =50;
        var ctx1 = canvas1.getContext("2d");
        ctx1.drawImage(canvas3, Xmin-15, Ymin-15, w + 30 , h + 30 , 0, 0, 50, 50);
        canvas1.style.backgroundColor = "white";
        var image = canvas1.toDataURL({format: 'png', multiplier: 4});
        }
        else {
        var image = canvas.toDataURL({format: 'png', multiplier: 4});
        } 
        console.log(image);
        image = encodeURIComponent(image);
        //image = image.replace('data:image/png;base64,','');
        
        var xhttp = new XMLHttpRequest();
        //let url ="";
        xhttp.open("GET",'ajax?algo='+algo+'&img='+image);
        //xhttp.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
        xhttp.send();
        if (clicked){
            container.style.transition = "all 2s";
            container.style.gridTemplateColumns = '1fr 1fr 2fr';
            container.style.gridTemplateRows = '';

        chhart.style.opacity = '1';
        chhart.style.transition = "all 3s";
        chhart.style.gridRow = '1';
        chhart.style.gridColumn = '3/4';
        clicked = false;
        }
        xhttp.onreadystatechange = function() {
        //console.log(`${this.responseText}`);
        if (this.readyState == 4 && this.status == 200) {
            let data = JSON.parse(this.responseText);
            let array = JSON.parse(data['array']);
            let top3 = JSON.parse(data['top3']);
            
            let ctx = document.querySelector('#chart').getContext('2d');
            //ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            //document.querySelector('#chart').textContent = '';
            //ctx.canvas.width = 300;
            //ctx.canvas.height = 300;
            let myBarChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels : [letters[array[0]],letters[array[1]],letters[array[2]]], 
                datasets:[
                    {
                        label:'أقرب ثلاث حروف' ,
                        data: top3,
                        backgroundColor: [
                            `rgb(${Math.random()*255},${Math.random()*255},${Math.random()*255})`,
                            `rgb(${Math.random()*255},${Math.random()*255},${Math.random()*255})`,
                            `rgb(${Math.random()*255},${Math.random()*255},${Math.random()*255})`,
        ],
                    }
                ]
                        },
            options: {
                responsive:false,
                scales: {
        yAxes: [{
            ticks: {
                fontSize: 20
            }
        }],
        xAxes: [{
            ticks: {
                fontSize: 20
            }
        }]
    }
            }
});
            console.log( top3[0]);
            console.log( array[0]);
            document.querySelector("#res").innerHTML = letters[array[0]];
           }
        else {
            console.log(`${this.status}`);
        }
         };
    });

del.addEventListener('click' ,clearArea);
correct.addEventListener('click' , function(){
var person = prompt("Please enter the correct answer:");

if (person == null || person == "") {
    person = prompt("you didn't write anything! Please enter the correct answer: ");
} else {
    var image2 = canvas.toDataURL({format: 'png', multiplier: 4});
        console.log(' hhhhhhhere');
        image2 = encodeURIComponent(image2);
        console.log((person));
        var char = letters.indexOf(person).toString();
        console.log('hohoh');
        var xhttp1 = new XMLHttpRequest();
        //let url ="";
        xhttp1.open("GET",'update?char='+char+'&img='+image2);
        xhttp1.send();
        
        xhttp1.onreadystatechange = function() {
        //console.log(`${this.responseText}`);
        if (this.readyState == 4 && this.status == 200) {
            alert('done!');
            
           }
           else {
               console.log('a problem!');
           }           }}
});
            
        
    
