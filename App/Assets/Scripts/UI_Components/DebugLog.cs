using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public enum LogType
{
    DefaultLog,
    WebCamText,
    NativeCamera,
    NativeGallery,
    Result
}

public class DebugLog : MonoBehaviour
{
    public TextMeshProUGUI Log;
    private static DebugLog instance = null;
    private LogType currentLogType = LogType.DefaultLog;
    
    // Start is called before the first frame update
    private void Awake() 
    {
        createInstance();
    }

    //Create Singleton to use anywhere
    private void createInstance()
    {
        instance = this;
    }

    public static DebugLog getInstance()
    {
        return instance;
    }

    public void updateLog(LogType LogType = LogType.DefaultLog, string content = "", bool IsAdding = true)
    {
        string _LogType = currentLogType == LogType? "" : LogType.ToString();
        currentLogType = LogType;

        if(IsAdding)
        {
            Log.text += System.Environment.NewLine + _LogType + ": " + content;
            return;
        }
        Log.text = currentLogType + ": " + content;
    }

    void Start() 
    {
        updateLog(LogType.DefaultLog, "Debug will be printed here", false);
    }
}
