document.getElementById("tireForm").addEventListener("submit", async function(e){

e.preventDefault()

const roadType = document.getElementById("roadType").value
const loadKg = parseInt(document.getElementById("loadKg").value)
const axles = parseInt(document.getElementById("axles").value)
const climate = document.getElementById("climate").value

const temperature = parseFloat(document.getElementById("temperature").value)
const speed = parseFloat(document.getElementById("speed").value)
const tireAge = parseFloat(document.getElementById("tireAge").value)
const wearLevel = document.getElementById("wearLevel").value

try{

const response = await fetch("http://127.0.0.1:5000/predict",{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify({
roadType,
loadKg,
axles,
climate,
temperature,
speed,
tireAge,
wearLevel
})
})

const data = await response.json()

console.log("Prediction:",data)

document.getElementById("steerTire").innerText = data.steerTire
document.getElementById("driveTire").innerText = data.driveTire
document.getElementById("trailerTire").innerText = data.trailerTire

document.getElementById("pressureValue").innerText =
data.pressure + " PSI"

document.getElementById("failureRisk").innerText =
data.failureRisk

document.getElementById("loadPerAxle").innerText =
data.loadPerAxle + " kg"

document.getElementById("safetyAdvice").innerText =
data.safetyAdvice

document.getElementById("resultSection").classList.remove("hidden")

}
catch(err){

console.error(err)
alert("Error displaying prediction (check console)")

}

})