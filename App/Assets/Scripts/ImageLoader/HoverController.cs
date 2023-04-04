using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class HoverController : MonoBehaviour
{
    public Button ToggleBtn;
    public TextMeshProUGUI LabelBtn;
    public float minX;
    public float maxX;
    public float minY;
    public float maxY;

    private bool EnableTouchInput = false;
    private Vector2 position = Vector2.zero;
    private bool allowControl = true;
    
    public void setHoverParam(float _maxX, float _minX, float _maxY, float _minY)
    {
        maxX = _maxX;
        minX = _minX;
        maxY = _maxY;
        minY = _minY;
    }

    public void setAllowMovement(bool isOn)
    {
        allowControl = isOn;
    }

    public void ToggleAllowControl()
    {
        allowControl = !allowControl;
        LabelBtn.text = allowControl? "Move" : "OK";
    }
    
    private void Awake() 
    {
        // #if UNITY_ANDROID || UNITY_IOS
        //     EnableTouchInput = true;
        // #endif

        // #if UNITY_STANDALONE || UNITY_WEBGL || UNITY_EDITOR
        //     EnableTouchInput = false;
        // #endif
    }

    void Start()
    {
        
    }

    private void OnEnable() 
    {
        allowControl = true;
        LabelBtn.text = "Move";
    }

    private void OnDisable() 
    {
        allowControl = false;
        LabelBtn.text = "OK";
    }

    void Update()
    {
        if(allowControl)
        {
            InputProcessing();
            HoverMovement();
        }   
    }

    private void InputProcessing()
    {
        if(EnableTouchInput)
        {
            if(Input.touchCount > 0)
            {
                Touch touch = Input.GetTouch(0);
                position = touch.position;
                Debug.Log("Position" + position);
            }
        }
        else
        {
            if(Input.GetButtonDown("Fire1"))
            {
                position = Input.mousePosition;
                Debug.Log("Position" + position);
            }
        }

        if(position != (Vector2)transform.position)
        {
            transform.position = position;
        }
    }

    private void HoverMovement()
    {
        var checkPosX = transform.GetComponent<RectTransform>().anchoredPosition.x;
        var checkPosY = transform.GetComponent<RectTransform>().anchoredPosition.y;
        if(checkPosX < minX)
        {
            checkPosX = minX;
        }
        else if(checkPosX > maxX)
        {
            checkPosX = maxX;
        }

        if(checkPosY < minY)
        {
            checkPosY = minY;
        }
        else if(checkPosY > maxY)
        {
            checkPosY = maxY;
        }
        transform.GetComponent<RectTransform>().anchoredPosition = new Vector2(checkPosX, checkPosY);
    }
}
