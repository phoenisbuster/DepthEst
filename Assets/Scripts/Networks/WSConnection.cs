using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using NativeWebSocket;

public class WSConnection : MonoBehaviour
{
    public string url = "ws://";
    public string address = "localhost:8080";
    private WebSocket websocket;

    private static WSConnection instance;

    private void Awake() 
    {
        instance = this;
    }

    public static WSConnection getInstance()
    {
        return instance;
    }

    private void OnDestroy() 
    {
        instance = null;
    }

    void Start()
    {
        
    }

    public IEnumerator ConnectWS()
    {
        yield return new WaitForSeconds(1);
        OpenWS();
    }

    public void changeAddress(string value, bool reconnect = false)
    {
        address = value;
        if(websocket != null)
            CloseWS();
        if(reconnect)
        {
            StartCoroutine(ConnectWS());
        }
    }

    public void DisconnectWS()
    {
        if(websocket != null)
            CloseWS();
    }

    private async void OpenWS()
    {
        DebugLog.getInstance().updateLog(LogType.DefaultLog, url+address, false);
        
        websocket = new WebSocket(url+address);

        websocket.OnOpen += () =>
        {
            Debug.Log("Connection open!");
        };

        websocket.OnError += (e) =>
        {
            Debug.Log("Error! " + e);
        };

        websocket.OnClose += (e) =>
        {
            Debug.Log("Connection closed!");
        };

        websocket.OnMessage += (bytes) =>
        {
            Debug.Log("OnMessage!");
            Debug.Log(bytes);

            // getting the message as a string
            // var message = System.Text.Encoding.UTF8.GetString(bytes);
            // Debug.Log("OnMessage! " + message);
        };

        // Keep sending messages at every 0.3s
        InvokeRepeating("SendWebSocketMessage", 0.0f, 0.3f);

        // waiting for messages
        await websocket.Connect();
    }

    void Update()
    {
        #if !UNITY_WEBGL || UNITY_EDITOR
            if(websocket != null)
                websocket.DispatchMessageQueue();
        #endif
    }

    async void SendWebSocketMessage()
    {
        if (websocket.State == WebSocketState.Open)
        {
            // Sending bytes
            await websocket.Send(new byte[] { 10, 20, 30 });

            // Sending plain text
            await websocket.SendText("plain text message");
        }
    }

    private async void CloseWS()
    {
        await websocket.Close();
    }

    private async void OnApplicationQuit()
    {
        if(websocket != null)
            await websocket.Close();
    } 
}
