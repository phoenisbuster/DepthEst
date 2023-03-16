using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UIManager : MonoBehaviour
{
    public GameObject SettingPanel;
    public bool SettingPanelIsOn = true;
    public bool firstInit = true;
    
    // Start is called before the first frame update
    void Start()
    {
        
    }

    private void OnEnable() 
    {
        PanelUp.OnClickSettingAction += ToggleSettingPanel;
    }

    private void OnDisable() 
    {
        PanelUp.OnClickSettingAction -= ToggleSettingPanel;
    }

    public void ToggleSettingPanel()
    {
        if(firstInit)
        {
            BuiltInCameraFunctions.getInstance().clickAccessCamera();
        }
        
        SettingPanelIsOn = !SettingPanelIsOn;
        SettingPanel.SetActive(SettingPanelIsOn);
        firstInit = false;
    }
}
