function generateGallery(){
    document.getElementById('submitbtn').disabled = true;
    x = document.getElementById('data').value;
    eel.test(x)(function(ret){

        for (i = 0; i < ret.length; i++) {
            console.log("../."+ret[i])
            var img = document.createElement("img");
            img.src = "../." + ret[i];
            img.width = 300;
            img.height= 200;
            img.style.marginLeft = '25px';
            img.style.marginBottom = '25px';
            var src = document.getElementById("gallery-div");
            src.appendChild(img);
        }

    })
}

function clearGallery(){
    document.getElementById('submitbtn').disabled = false;
    document.getElementById('gallery-div').innerHTML = " ";
}