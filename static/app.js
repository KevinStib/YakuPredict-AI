const form = document.getElementById('prediction-form');
const riskScore = document.getElementById('risk-score');
const riskBar = document.getElementById('risk-bar');
const riskLevel = document.getElementById('risk-level');
const supervisedScore = document.getElementById('supervised-score');
const anomalyScore = document.getElementById('anomaly-score');
const decisionCaption = document.getElementById('decision-caption');
const emptyState = document.getElementById('recommendation-state');
const resultState = document.getElementById('recommendation-result');
const decisionLevel = document.getElementById('decision-level');
const recommendationText = document.getElementById('recommendation-text');
const rfBreakdown = document.getElementById('rf-breakdown');
const ifBreakdown = document.getElementById('if-breakdown');
const historyBody = document.getElementById('history-body');
const submitButton = document.getElementById('submit-btn');

const presets = {
  normal: {flow_m3s:78,power_mw:72,vibration_mms:2.2,bearing_temp_c:66,oil_temp_c:52,hydraulic_pressure_bar:74,wicket_gate_pct:75,stator_current_a:1360,ambient_temp_c:22},
  watch: {flow_m3s:76,power_mw:69,vibration_mms:4.1,bearing_temp_c:76,oil_temp_c:59,hydraulic_pressure_bar:70,wicket_gate_pct:73,stator_current_a:1420,ambient_temp_c:23},
  critical: {flow_m3s:80,power_mw:67,vibration_mms:6.4,bearing_temp_c:88,oil_temp_c:67,hydraulic_pressure_bar:65,wicket_gate_pct:78,stator_current_a:1510,ambient_temp_c:23}
};

document.querySelectorAll('[data-preset]').forEach(button => {
  button.addEventListener('click', () => {
    const values = presets[button.dataset.preset];
    Object.entries(values).forEach(([key,value]) => form.elements[key].value = value);
  });
});

function levelClass(level){return ({'BAJO':'low','MEDIO':'medium','ALTO':'high','CRÍTICO':'critical'})[level] || 'neutral'}

async function loadHistory(){
  try{
    const response = await fetch('/history?limit=8');
    const items = await response.json();
    if(!items.length){historyBody.innerHTML='<tr><td colspan="6" class="table-empty">Sin registros</td></tr>';return;}
    historyBody.innerHTML = items.map(item => `
      <tr>
        <td>#${item.id}</td>
        <td>${new Date(item.created_at).toISOString().replace('T',' ').slice(0,19)}</td>
        <td class="risk-chip">${item.risk_percent.toFixed(2)} %</td>
        <td><span class="level-chip ${item.level.toLowerCase()}">${item.level}</span></td>
        <td>${(item.supervised_probability*100).toFixed(1)} %</td>
        <td>${(item.anomaly_probability*100).toFixed(1)} %</td>
      </tr>`).join('');
  } catch(error){historyBody.innerHTML='<tr><td colspan="6" class="table-empty">No fue posible cargar el historial</td></tr>'}
}

document.getElementById('refresh-history').addEventListener('click', loadHistory);

form.addEventListener('submit', async event => {
  event.preventDefault();
  const data = {};
  new FormData(form).forEach((value,key) => data[key] = Number(value));
  submitButton.disabled = true;
  submitButton.textContent = 'Analizando...';
  try{
    const response = await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const result = await response.json();
    if(!response.ok) throw new Error(result.detail || 'Error de inferencia');
    riskScore.textContent = `${result.risk_percent.toFixed(2)} %`;
    riskBar.style.width = `${Math.max(1,result.risk_percent)}%`;
    riskLevel.textContent = result.level;
    riskLevel.className = `level-badge ${levelClass(result.level)}`;
    supervisedScore.textContent = `${(result.supervised_probability*100).toFixed(1)} %`;
    anomalyScore.textContent = `${(result.anomaly_probability*100).toFixed(1)} %`;
    decisionCaption.textContent = `Registro #${result.id}`;
    emptyState.classList.add('hidden');
    resultState.classList.remove('hidden');
    decisionLevel.textContent = result.level;
    recommendationText.textContent = result.recommendation;
    rfBreakdown.textContent = `${(result.supervised_probability*100).toFixed(1)} %`;
    ifBreakdown.textContent = `${(result.anomaly_probability*100).toFixed(1)} %`;
    await loadHistory();
  }catch(error){
    alert(error.message);
  }finally{
    submitButton.disabled = false;
    submitButton.textContent = 'Analizar condición';
  }
});

loadHistory();
