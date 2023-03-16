using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using DG.Tweening;
using TMPro;

public class UIManager : MonoBehaviour
{
    public GameObject BackGround;
    public ImageLoader ImageLoaderScript;
    public GameObject Panel_Up;
    public GameObject Panel_Down;
    public GameObject SettingPanel;
    public bool SettingPanelIsOn = true;
    public bool firstInit = true;
    public float width = 0;
    public float height = 0;
    
    // Start is called before the first frame update
    void Start()
    {
        width = transform.GetComponent<RectTransform>().rect.width;
        height = transform.GetComponent<RectTransform>().rect.height;
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
            //BuiltInCameraFunctions.getInstance().clickAccessCamera();
        }
        firstInit = false;

        if(SettingPanelIsOn)
        {
            SettingPanel.transform.DOScale(new Vector3(0,0,0), 0.25f).SetLink(gameObject).OnComplete(()=>
            {
                SettingPanel.SetActive(false);
            });
        }
        else
        {
            SettingPanel.SetActive(true);
            SettingPanel.transform.DOScale(new Vector3(1,1,1), 0.25f).SetLink(gameObject).OnComplete(()=>
            {
                
            });
        }
        SettingPanelIsOn = !SettingPanelIsOn;
    }
}
