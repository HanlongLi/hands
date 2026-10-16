(function () {
  const buttons = [...document.querySelectorAll('[data-tab]')];
  window.PanelWorkspace = {
    current: 'mapping',
    select(name) {
      if (!buttons.some(button => button.dataset.tab === name)) return;
      if (name !== this.current) {
        send({type:'teleop_arm',on:false});
        drivingTeleop = false;
        if (name !== 'mapping' && window.DemoLab) window.DemoLab.pause();
      }
      for (const button of buttons) {
        const selected = button.dataset.tab === name;
        button.setAttribute('aria-selected',String(selected));button.tabIndex=selected?0:-1;
        document.getElementById(button.getAttribute('aria-controls')).hidden=!selected;
      }
      this.current=name;history.replaceState(null,'',`#${name}`);
      window.dispatchEvent(new Event('resize'));
    },
  };
  buttons.forEach((button,index) => {
    button.onclick=()=>PanelWorkspace.select(button.dataset.tab);
    button.onkeydown=event=>{
      if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
      event.preventDefault();
      let next=event.key==='Home'?0:event.key==='End'?buttons.length-1:(index+(event.key==='ArrowRight'?1:-1)+buttons.length)%buttons.length;
      PanelWorkspace.select(buttons[next].dataset.tab);buttons[next].focus();
    };
  });
  PanelWorkspace.select(location.hash.slice(1)||'mapping');
})();
