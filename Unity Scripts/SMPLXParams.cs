using UnityEngine;
using System.Collections.Generic;
[System.Serializable]
public class SMPLXParams
{
    public float[][] pose;
    public float[] shape;
    public float[] expression;
    public float[] location;
    public float distance;
    public float[] rotation;
    public float[] positionOffset;
    public float[] rotationOffset;
    public CameraParams camera;
    public Vector3 locationV
    {
        get { return new Vector3(location[0], location[1], 0); }   // get method
        //set { name = value; }  // set method
    }
    public Vector3 rotationV
    {
        get { return new Vector3(rotation[0], rotation[1], rotation[2]); }   // get method
        //set { name = value; }  // set method
    }
    public Vector3 positionOffsetV
    {
        get { return new Vector3(positionOffset[0], positionOffset[1], positionOffset[2]); }   // get method
        //set { name = value; }  // set method
    }
    public Vector3 rotationOffsetV
    {
        get { return new Vector3(rotationOffset[0], rotationOffset[1], rotationOffset[2]); }   // get method
        //set { name = value; }  // set method
    }
}