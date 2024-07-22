using UnityEngine;
using Newtonsoft.Json;
using System.Collections.Generic;

public class SMPLXAligner : MonoBehaviour
{
    // Propiedades públicas
    public string jsonFilePath;
    public Texture2D backgroundImage;
    public GameObject smplxPrefab;
    public Camera alignmentCamera;
    public bool showJoints2D = true;
    public float joint2DSize = 5f; // Tamaño de los círculos en píxeles
    public Color joint2DColor = Color.green;
    public bool showJoints3D = true;
    public Color joint3DColor = Color.blue;
    public float joint3DScale = 0.02f;
    

    // Propiedades privadas
    private string[] SMPLX_JOINT_NAMES = new string[] { "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_index1", "left_index2", "left_index3", "left_middle1", "left_middle2", "left_middle3", "left_pinky1", "left_pinky2", "left_pinky3", "left_ring1", "left_ring2", "left_ring3", "left_thumb1", "left_thumb2", "left_thumb3", "right_index1", "right_index2", "right_index3", "right_middle1", "right_middle2", "right_middle3", "right_pinky1", "right_pinky2", "right_pinky3", "right_ring1", "right_ring2", "right_ring3", "right_thumb1", "right_thumb2", "right_thumb3", "jaw"};
    private string[] ALL_JOINT_NAMES = { "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_index1", "left_index2", "left_index3", "left_middle1", "left_middle2", "left_middle3", "left_pinky1", "left_pinky2", "left_pinky3", "left_ring1", "left_ring2", "left_ring3", "left_thumb1", "left_thumb2", "left_thumb3", "right_index1", "right_index2", "right_index3", "right_middle1", "right_middle2", "right_middle3", "right_pinky1", "right_pinky2", "right_pinky3", "right_ring1", "right_ring2", "right_ring3", "right_thumb1", "right_thumb2", "right_thumb3", "jaw", "left_eye_smplhf", "right_eye_smplhf", "nose", "right_eye", "left_eye", "right_ear", "left_ear", "left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel", "left_thumb", "left_index", "left_middle", "left_ring", "left_pinky", "right_thumb", "right_index", "right_middle", "right_ring", "right_pinky", "right_eye_brow1", "right_eye_brow2", "right_eye_brow3", "right_eye_brow4", "right_eye_brow5", "left_eye_brow5", "left_eye_brow4", "left_eye_brow3", "left_eye_brow2", "left_eye_brow1", "nose1", "nose2", "nose3", "nose4", "right_nose_2", "right_nose_1", "nose_middle", "left_nose_1", "left_nose_2", "right_eye1", "right_eye2", "right_eye3", "right_eye4", "right_eye5", "right_eye6", "left_eye4", "left_eye3", "left_eye2", "left_eye1", "left_eye6", "left_eye5", "right_mouth_1", "right_mouth_2", "right_mouth_3", "mouth_top", "left_mouth_3", "left_mouth_2", "left_mouth_1", "left_mouth_5", "left_mouth_4", "mouth_bottom", "right_mouth_4", "right_mouth_5", "right_lip_1", "right_lip_2", "lip_top", "left_lip_2", "left_lip_1", "left_lip_3", "lip_bottom", "right_lip_3", "right_contour_1", "right_contour_2", "right_contour_3", "right_contour_4", "right_contour_5", "right_contour_6", "right_contour_7", "right_contour_8", "contour_middle", "left_contour_8", "left_contour_7", "left_contour_6", "left_contour_5", "left_contour_4", "left_contour_3", "left_contour_2", "left_contour_1" };
    private Vector2[] joints2DScreenPositions; // Array para almacenar las posiciones de los joints en la pantalla
    private SMPLXParams parameters;
    private GameObject joint3dParentGO;
    private List<GameObject> jointObjects = new List<GameObject>();

    void Start()
    {
        parameters = JSONReader.ReadJSONFile(jsonFilePath);
        if (parameters != null)
        {
            SetupCamera();
            AlignSMPLX();
            SetupBackgroundImage();
        }
    }

    void Update()
    {
        if (showJoints3D != joint3dParentGO.activeSelf) joint3dParentGO.SetActive(showJoints3D);
    }

    void OnGUI()
    {
        if (joints2DScreenPositions == null) return;

        if (showJoints2D)
        {
            // Crear un estilo para los círculos
            GUIStyle circleStyle = new GUIStyle();
            Texture2D texture = new Texture2D(1, 1);
            texture.SetPixel(0, 0, joint2DColor);
            texture.Apply();
            circleStyle.normal.background = texture;

            // Dibujar cada joint como un círculo rojo
            for (int i = 0; i < joints2DScreenPositions.Length; i++)
            {
                Vector2 pos = joints2DScreenPositions[i];
                
                // Invertir la coordenada Y para la GUI de Unity
                pos.y = Screen.height - pos.y;
                
                Rect rect = new Rect(pos.x - joint2DSize / 2, pos.y - joint2DSize / 2, joint2DSize, joint2DSize);
                GUI.Box(rect, GUIContent.none, circleStyle);
                
                // Opcionalmente, puedes dibujar el índice del joint para depuración
                // GUI.Label(new Rect(pos.x, pos.y, 20, 20), i.ToString());
            }
        }
    }

    private void SetupCamera()
    {
        float fx = parameters.camera_intrinsics[0][0];
        float fy = parameters.camera_intrinsics[1][1];
        float cx = parameters.camera_intrinsics[0][2];
        float cy = parameters.camera_intrinsics[1][2];

        float fovY = 2 * Mathf.Atan(parameters.resized_height / (2 * fy)) * Mathf.Rad2Deg;
        alignmentCamera.fieldOfView = fovY;

        float aspect = (float)parameters.resized_width / parameters.resized_height;
        alignmentCamera.aspect = aspect;

    }

    private void AlignSMPLX()
    {
        foreach (HumanParams human in parameters.humans)
        {
            // Instanciar objetos necesarios
            GameObject smplxInstance = Instantiate(smplxPrefab);
            SMPLX smplxComponent = smplxInstance.GetComponent<SMPLX>();
            
            // Aplicar la pose, la forma y las expresiones al SMPL-X
            if (human.rotation_vector != null) ApplyPose(smplxComponent, human.rotation_vector);
            if (human.shape != null) ApplyShape(smplxComponent, human.shape);
            if (human.expression != null) ApplyExpression(smplxComponent, human.expression);

            // Rotar en 180° para que quede bien alineado
            smplxInstance.transform.rotation *= Quaternion.Euler(0, 0, 180);
            
            // Posicionar smpl segun el primary keypoint (head)
            if ((human.location != null) && (human.translation != null) && (human.translation_pelvis != null) && (parameters.resized_width != 0) && (parameters.resized_height != 0) && (parameters.checkpoint_resolution != 0)) 
            {
                AdjustScale(smplxInstance, human.translation, human.translation_pelvis);
                AlignWithHead3D(smplxInstance, human.translation, parameters.resized_width, parameters.resized_height, parameters.checkpoint_resolution);
                AdjustRotation(smplxInstance, human.translation, human.translation_pelvis);
            }
            
            // Dibujar los Joints2D
            if ((human.joints_2d != null)&&(human.translation_pelvis != null) && (parameters.resized_width != 0) && (parameters.resized_height != 0) && (parameters.checkpoint_resolution != 0)) {
                CalculateJoints2DScreenPositions(human.joints_2d, parameters.resized_width, parameters.resized_height, parameters.checkpoint_resolution);
            }
            
            // Dibujar los Joints3D
            if (human.joints_3d != null)
            {
                joint3dParentGO = new GameObject("Joints3D");
                Transform joint3dParent = joint3dParentGO.transform;
                CreateJoints3D(human.joints_3d, joint3dParent);
            }
        }
    }

    private void ApplyPose(SMPLX smplxMannequin, float[][] newPose) 
    {
        if (newPose.Length != SMPLX_JOINT_NAMES.Length)
        {
            Debug.LogError($"Incorrect number of pose values: {newPose.Length}, expected: {SMPLX_JOINT_NAMES.Length}");
            return;
        }

        for (int i = 0; i < newPose.Length; i++)
        {
            string jointName = SMPLX_JOINT_NAMES[i];
            float rodX = newPose[i][0];
            float rodY = newPose[i][1];
            float rodZ = newPose[i][2];

            Quaternion rotation = SMPLX.QuatFromRodrigues(rodX, rodY, rodZ);
            smplxMannequin.SetLocalJointRotation(jointName, rotation);
        }

        smplxMannequin.EnablePoseCorrectives(true);
        Debug.Log("Pose applied successfully.");
    }

    private void ApplyShape(SMPLX smplxMannequin, float[] newBetas)
    {
        if (newBetas.Length != SMPLX.NUM_BETAS)
        {
            Debug.LogError($"Incorrect number of betas: {newBetas.Length}, expected: {SMPLX.NUM_BETAS}");
            return;
        }
        
        for (int i = 0; i < SMPLX.NUM_BETAS; i++)
        {
            smplxMannequin.betas[i] = newBetas[i];
        }
        smplxMannequin.SetBetaShapes();
        Debug.Log("Shape parameters applied successfully.");
    }

    private void ApplyExpression(SMPLX smplxMannequin, float[] newExpression) 
    {
        if (newExpression.Length != SMPLX.NUM_EXPRESSIONS)
        {
            Debug.LogError($"Incorrect number of expressions: {newExpression.Length}, expected: {SMPLX.NUM_EXPRESSIONS}");
            return;
        }
        
        for (int i = 0; i < SMPLX.NUM_EXPRESSIONS; i++)
        {
            smplxMannequin.expressions[i] = newExpression[i];
        }
        smplxMannequin.SetExpressions();
        Debug.Log("Expression parameters applied successfully.");
    }

    private void AdjustScale(GameObject smplxGO, float[] joint3DHead, float[][] joint3DPelvis)
    {
        Vector3 pelvis3D = new Vector3(joint3DPelvis[0][0], -joint3DPelvis[0][1], joint3DPelvis[0][2]);
        Vector3 head3D = new Vector3(joint3DHead[0], -joint3DHead[1], joint3DHead[2]);

        // Calcular factor de escala
        float targetDistance = Vector3.Distance(head3D, pelvis3D);
        Vector3 smplxHead = smplxGO.GetComponent<SMPLX>().JointTransforms[SMPLX_JOINT_NAMES[15]].position;
        Vector3 smplxPelvis = smplxGO.GetComponent<SMPLX>().JointTransforms[SMPLX_JOINT_NAMES[0]].position;
        float currentDistance = Vector3.Distance(smplxHead, smplxPelvis);
        float scaleFactor = targetDistance / currentDistance;

        // Aplicar escala al SMPL-X
        smplxGO.transform.localScale *= scaleFactor;
    }

    private void AlignWithHead3D(GameObject smplxInstance, float[] joint3DHead, int image_width, int image_height, int model_img_size)
    {
        Transform headTransform = smplxInstance.GetComponent<SMPLX>().JointTransforms[SMPLX_JOINT_NAMES[15]];
        Vector3 relativePositionHead = smplxInstance.transform.position - headTransform.position;
        
        Vector3 head3DWorld = new Vector3(joint3DHead[0], -joint3DHead[1], joint3DHead[2]);

        // Ajustar la posición del modelo SMPLX
        Vector3 offset = head3DWorld - smplxInstance.transform.position;
        smplxInstance.transform.position += offset + relativePositionHead;

        Debug.Log($"offset: {offset}, relativePositionHead: {relativePositionHead}, smplxInstance.transform.position: {smplxInstance.transform.position + offset + relativePositionHead}, 3D World: {head3DWorld}, headTransform.position: {headTransform.position}");
    }

    private void AdjustRotation(GameObject smplxGO, float[] joint3DHead, float[][] joint3DPelvis)
    {
        Vector3 smplxHead = smplxGO.GetComponent<SMPLX>().JointTransforms[SMPLX_JOINT_NAMES[15]].position;
        Vector3 smplxPelvis = smplxGO.GetComponent<SMPLX>().JointTransforms[SMPLX_JOINT_NAMES[0]].position;

        Vector3 pelvis3D = new Vector3(joint3DPelvis[0][0], -joint3DPelvis[0][1], joint3DPelvis[0][2]);
        Vector3 head3D = new Vector3(joint3DHead[0], -joint3DHead[1], joint3DHead[2]);
        
        // Dejar la cabeza como el pivote para rotar el SMPL-X
        GameObject smplxPivotGO = new GameObject();
        Transform smplxPivot = smplxPivotGO.transform;
        smplxPivot.position = new Vector3(smplxHead.x, smplxHead.y, smplxHead.z);
        smplxGO.transform.parent = smplxPivot;

        // Calcular rotación
        Vector3 currentPelvisDirection = (smplxPelvis - smplxHead).normalized;
        Vector3 targetPelvisDirection = (pelvis3D - head3D).normalized;

        Quaternion additionalRotation = Quaternion.FromToRotation(currentPelvisDirection, targetPelvisDirection);
        Debug.Log($"currentPelvisDirection: {currentPelvisDirection}, targetPelvisDirection: {targetPelvisDirection}, additionalRotation: {additionalRotation}");

        // Aplicar rotación
        smplxPivot.rotation *= additionalRotation;
        Debug.Log($"smplxPivot.rotation después: {smplxPivot.eulerAngles}");

        // Borrar el pivote
        smplxGO.transform.parent = null;

        // Verificar alineación
        Vector3 finalHeadPosition = smplxGO.GetComponent<SMPLX>().JointTransforms[SMPLX_JOINT_NAMES[15]].position;
        Vector3 finalPelvisPosition = smplxGO.GetComponent<SMPLX>().JointTransforms[SMPLX_JOINT_NAMES[0]].position;
        Debug.Log($"Head Target: {head3D}, Actual: {finalHeadPosition}");
        Debug.Log($"Pelvis Target: {pelvis3D}, Actual: {finalPelvisPosition}");
        
        // Destruir los GameObjects creados
        Destroy(smplxPivotGO);
    }

    private void SetupBackgroundImage()
    {
        GameObject quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        quad.name = "Plano Imagen Original";
        quad.transform.position = new Vector3(0, 0, 10);
        
        float aspectRatio = (float)backgroundImage.width / backgroundImage.height;
        //float aspectRatio = 9f/16f;
        float quadHeight = 2f * Mathf.Tan(alignmentCamera.fieldOfView * 0.5f * Mathf.Deg2Rad) * quad.transform.position.z;
        float quadWidth = quadHeight * aspectRatio;

        //Debug.Log($"backgroundImage.width: {backgroundImage.width}, backgroundImage.height: {backgroundImage.height}, aspectRatio: {aspectRatio}");
        
        quad.transform.localScale = new Vector3(quadWidth, quadHeight, 1);
        
        Material mat = new Material(Shader.Find("Unlit/Texture"));
        mat.mainTexture = backgroundImage;
        quad.GetComponent<Renderer>().material = mat;
    }

    private void CalculateJoints2DScreenPositions(float[][] joints_2d, int image_width, int image_height, int model_img_size)
    {
        joints2DScreenPositions = new Vector2[joints_2d.Length];

        // Calcular el factor de escala
        float scaleX = (float)Screen.width / image_width;
        float scaleY = (float)Screen.height / image_height;

        for (int i = 0; i < joints_2d.Length; i++)
        {
            joints2DScreenPositions[i] = Updated2DCoordinatesWithScale(joints_2d[i][0], joints_2d[i][1], image_width, image_height, scaleX, scaleY, model_img_size);
            //if (i == 0) Debug.Log($"Pelvis joints2DScreenPositions[{i}]: {joints2DScreenPositions[i]}");
        }
    }

    private Vector2 Updated2DCoordinates(float x_2d, float y_2d, int image_width, int image_height, int model_img_size)
    {
        // Calcular el factor de escala
        float scaleX = (float)Screen.width / image_width;
        float scaleY = (float)Screen.height / image_height;
            
        return Updated2DCoordinatesWithScale(x_2d, y_2d, image_width, image_height, scaleX, scaleY, model_img_size);
    }

    private Vector2 Updated2DCoordinatesWithScale(float x_2d, float y_2d, float image_width, int image_height, float scaleX, float scaleY, int model_img_size)
    {
        float newX = (x_2d - (model_img_size - image_width)/2) * scaleX;
        float newY = (image_height - y_2d) * scaleY;
        
        return new Vector2(newX, newY);
    }

    private void CreateJoints3D(float[][] joints3D, Transform parent)
    {
        parent.position = new Vector3(joints3D[0][0], -joints3D[0][1], joints3D[0][2]);
        for (int i=0; i < joints3D.Length; i++)
        {
            float[] joint = joints3D[i];
            GameObject jointObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            jointObject.transform.parent = parent;
            jointObject.transform.position = new Vector3(joint[0], -joint[1], joint[2]);
            jointObject.transform.localScale = Vector3.one * joint3DScale;
            jointObjects.Add(jointObject);

            jointObject.name = i + "_" + ALL_JOINT_NAMES[i];
            // Eliminar el collider
            Destroy(jointObject.GetComponent<SphereCollider>());

            // Cambiar el color de la esfera
            Renderer sphereRenderer = jointObject.GetComponent<Renderer>();
            if (sphereRenderer != null)
            {
                // Cambiar el color del material
                sphereRenderer.material.color = joint3DColor;
            }
        }
        Debug.Log("Joints3D created successfully.");

        AlignWithPelvis2D(parent);
    }

    private void AlignWithPelvis2D(Transform pelvis3D)
    {
        // Obtener las coordenadas 2D de la pelvis (asumiendo que es el primer joint)
        Vector2 pelvis2D = new Vector2 (joints2DScreenPositions[0].x, joints2DScreenPositions[0].y);
        Vector3 newPelvis3DPosition = alignmentCamera.ScreenToWorldPoint(new Vector3(pelvis2D.x, pelvis2D.y, pelvis3D.position.z));
        pelvis3D.position = newPelvis3DPosition;
    }

    public void Setjoint3DScale(float scale)
    {
        joint3DScale = scale;
        foreach (GameObject joint in jointObjects)
        {
            joint.transform.localScale = Vector3.one * joint3DScale;
        }
    }
}