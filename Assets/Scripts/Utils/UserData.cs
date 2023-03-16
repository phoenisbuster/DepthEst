using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UserData
{
    private static string ServerAddress = "ServerAddress";
    private static string ResolutionOption = "ResolutionOption";

    public static void SaveServerAddress(string value = "")
    {
        PlayerPrefs.SetString(ServerAddress, value);
    }

    public static void SaveResolutionOption(int value = 0)
    {
        PlayerPrefs.SetInt(ResolutionOption, value);
    }

    public static string GetServerAddress()
    {
        return PlayerPrefs.GetString(ServerAddress, "");
    }

    public static int GetResolutionOption()
    {
        return PlayerPrefs.GetInt(ResolutionOption, 0);
    }

    public static void DeleteServerAddress()
    {
        PlayerPrefs.DeleteKey(ServerAddress);
    }

    public static void DeleteResolutionOption()
    {
        PlayerPrefs.DeleteKey(ServerAddress);
    }

    public static void DeleteAllUserData(string key)
    {
        PlayerPrefs.DeleteAll();
    }
}

class SendData
{
    public byte[] ImageData;
    public float ImageWidth;
    public float ImageHeight;
    public float PosX;
    public float PosY;
    public SendData(byte[] _ImageData, float _ImageWidth, float _ImageHeight, float _PosX, float _PosY)
    {
        ImageData = _ImageData;
        ImageWidth = _ImageWidth;
        ImageHeight = _ImageHeight;
        PosX = _PosX;
        PosY = _PosY;
    }
}

class ReceiveData
{
    public byte[] ImageData;
    public float depth;
    public ReceiveData(byte[] _ImageData, float _depth)
    {
        ImageData = _ImageData;
        depth = _depth;
    }
}
