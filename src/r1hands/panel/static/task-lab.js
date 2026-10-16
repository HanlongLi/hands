(function () {
  const $=id=>document.getElementById(id);
  const names=['Thumb','Thumb rot.','Index','Middle','Ring','Pinky'];
  let target=[0,0,0,0,0,0],task=null,inFlight=false,lastSavedId=null;
  async function action(name,extra={}) {
    $('task-error').textContent='';
    try {
      const response=await fetch('/tasks/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:name,...extra})});
      if(!response.ok)throw new Error(await response.text());
      task=await response.json();update(task);return task;
    } catch(error){$('task-error').textContent=error.message;}
  }
  $('task-sliders').innerHTML=names.map((name,i)=>`<label class="mapping-axis">${name}<input id="task-axis-${i}" aria-label="Task ${name}" type="range" min="0" max="1" step="0.01" value="0" disabled><output id="task-value-${i}">0.00</output></label>`).join('');
  function showTarget(){target.forEach((q,i)=>{$(`task-axis-${i}`).value=q;$(`task-value-${i}`).textContent=q.toFixed(2);});}
  for(let i=0;i<6;i++)$(`task-axis-${i}`).oninput=event=>{target[i]=Number(event.target.value);showTarget();action('target',{position:target});};
  $('task-start').onclick=async()=>{
    $('task-start').disabled=true;$('task-start').textContent='Starting MuJoCo…';
    await action('start');$('task-start').textContent=task?.ready?'Simulation running':'Start task simulation';$('task-start').disabled=!!task?.ready;
  };
  $('task-reset').onclick=async()=>{const result=await action('reset',{seed:Number($('task-seed').value)});if(result){target=result.observation.target.slice();showTarget();}};
  $('task-release').onclick=()=>action('release');$('task-hold').onclick=()=>action('hold');
  $('task-mapping').onclick=()=>action('mapping',{on:!task?.mapping_enabled});
  $('task-example').onclick=()=>{target=[0.734,0.382,0.323,0.541,0.78,0.626];showTarget();action('example');};
  $('task-open').onclick=()=>{target=[0,0,0,0,0,0];showTarget();action('target',{position:target});};
  $('task-begin').onclick=()=>action('begin',{operator_id:$('task-operator').value.trim()});
  $('task-finish').onclick=()=>action('finish');
  function update(value){
    task=value;
    const ready=!!task?.ready,obs=task?.observation,recording=!!task?.recording;
    for(const id of ['task-reset','task-release','task-hold','task-mapping','task-example','task-open','task-begin'])$(id).disabled=!ready||!!task?.locked;
    $('task-reset').disabled=!ready||recording;$('task-begin').disabled=!ready||recording||!!task?.locked;
    $('task-finish').disabled=!ready||!recording;
    for(let i=0;i<6;i++)$(`task-axis-${i}`).disabled=!ready||!!task?.locked;
    $('task-mapping').classList.toggle('on',!!task?.mapping_enabled);
    $('task-mapping').textContent=task?.mapping_enabled?'Stop mapping input':'Use live mapping input';
    if(task?.error)$('task-error').textContent=task.error;
    if(!obs)return;
    $('task-status').textContent=task.locked?'Stopped':obs.success?'Success':obs.support_removed?'Grasp test':'Support in place';
    $('task-hold-time').textContent=obs.hold_s.toFixed(2)+' s';
    $('task-contacts').textContent=obs.contacts.fingers.join(', ')||'None';
    $('task-force').textContent=obs.contacts.max_force_n.toFixed(2)+' N';
    $('task-record-status').textContent=recording?`Recording · ${task.recorded_steps} actions · ${(task.recorded_steps*0.05).toFixed(1)} s`:'Episode recorder ready · scene images and actions are synchronized';
    if(task.saved && task.saved.episode_id!==lastSavedId){
      lastSavedId=task.saved.episode_id;
      $('task-saved').replaceChildren();
      const link=document.createElement('a');link.href=`/tasks/episodes/${task.saved.episode_id}`;link.textContent=`Download episode · ${task.saved.success?'success':'failure'} · ${task.saved.steps} actions`;
      $('task-saved').appendChild(link);
    }
  }
  window.addEventListener('r1hands-state',({detail})=>update(detail.task));
  setInterval(async()=>{
    if(!task?.ready||inFlight||window.PanelWorkspace?.current!=='task')return;
    inFlight=true;
    try{
      const response=await fetch('/tasks/frame.png');if(!response.ok)return;
      const url=URL.createObjectURL(await response.blob()),image=$('task-image');
      const previous=image.src;image.src=url;image.hidden=false;$('task-placeholder').hidden=true;
      image.onload=()=>{if(previous.startsWith('blob:'))URL.revokeObjectURL(previous);};
    }catch(error){$('task-error').textContent='Task camera connection failed: '+error.message;}finally{inFlight=false;}
  },150);
})();
