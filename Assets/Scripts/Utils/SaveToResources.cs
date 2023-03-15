using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SaveToResources
{
    private static int FileCounter = 0;
    public static void Save(byte[] Content, string FolderName = "DefaultName", string FileType = "png")
    { 
        var dirPath = Application.persistentDataPath + "/DepthEstimation/" + FolderName;    
        if(!Directory.Exists(dirPath)) 
        {
            Directory.CreateDirectory(dirPath);
        }
        Debug.Log(dirPath);
        File.WriteAllBytes(dirPath + "/" + FileCounter + "." + FileType, Content);
        FileCounter++;
    }
}
