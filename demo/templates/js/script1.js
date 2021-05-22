window.addEventListener('DOMContentLoaded',()=>{

console.log('hello');
    const button = document.querySelector('#btn');
    button.addEventListener('click' , function(){
        const input = document.querySelector('#input').value;
        console.log(`${input}`);
        var xhttp = new XMLHttpRequest();
        let url ="{% url 'ajax' %}";
        xhttp.open("POST", 'http://127.0.0.1:8000/demo/ajax/');
        //xhttp.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
        xhttp.send();
        xhttp.onreadystatechange = function() {
        console.log(`${this.responseText}`);
        if (this.readyState == 4 && this.status == 200) {
            //let data = JSON.parse(this.responseText);
            document.querySelector("#res").innerHTML =  this.responseText;
           }
        else if (this.status == 0){
            console.log(`${this.readyState}`);
        }
        else {
            console.log(`${this.status}`);
        }
         };
    });
    
 });
