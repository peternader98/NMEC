let myalert = document.getElementById("alert")
let error  = document.getElementById("error-box")
let success  = document.getElementById("success-box")
let showdAlert = success

myalert.addEventListener("click", e=>{
    console.log(showdAlert)
    showdAlert.classList.remove('show')
    myalert.classList.remove('showAlert')
})

function showMessage(isSuccess , msg){
    if(isSuccess == true){
        showdAlert = success
    }
    else{
        showdAlert = error
    }	

    console.log(showdAlert)
    showdAlert.classList.add('show')
    showdAlert.getElementsByClassName("alert-message")[0].innerHTML = msg
    myalert.classList.add('showAlert')
}