using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IOU
{
    class GT_D_Cifti
    {
        public int GT_id;
        public int D_id;
        public int puan;
        public GT_D_Cifti(int GTid, int Did)
        {
            GT_id = GTid;
            D_id = Did;
        }
    }
    public class BoundingBox
    {
        /*				 ________________________
				  x0|y1                    x1|y1
					|                        |
					|      BoundingBox       |
					|                        |
				  x0|y0____________________x1|y0
	    */
        public int x0;
        public int y0;
        public int x1;
        public int y1;
        public string type;
        public BoundingBox(int x0, int y0, int x1, int y1, string type)
        {
            this.x0 = x0;
            this.y0 = y0;
            this.x1 = x1;
            this.y1 = y1;
            this.type = type;
        }
        public int Area()
        {
            return (x1 - x0) * (y1 - y0);
        }

        public int w
        {
            get
            {
                return x1 - x0;
            }
        }
        public int h
        {
            get
            {
                return y1 - y0;
            }
        }
    }
    class IOU
    {
        public static double Puanla(List<BoundingBox> GTler, List<BoundingBox> Dler)
        {

            List<KeyValuePair<GT_D_Cifti, double>> puanlar = Puanlar_matrisi_olustur(GTler, Dler);

            Dictionary<int, int> D_GT_eslesmeleri = new Dictionary<int, int>();
            List<int> Kul_GTler = new List<int>();
            List<int> Kul_Dler = new List<int>();
            double Toplam_puan = 0;
            for (int i = 0; i < puanlar.Count; i++)
            {
                KeyValuePair<GT_D_Cifti, double> GT_D_puan = puanlar[i];
                if (GT_D_puan.Value > 0)
                {
                    if (Kul_GTler.Contains(GT_D_puan.Key.GT_id) == false && Kul_Dler.Contains(GT_D_puan.Key.D_id) == false)
                    {
                        D_GT_eslesmeleri.Add(GT_D_puan.Key.D_id, GT_D_puan.Key.GT_id);
                        Kul_GTler.Add(GT_D_puan.Key.GT_id);
                        Kul_Dler.Add(GT_D_puan.Key.D_id);
                        Toplam_puan += Puan_hesapla(GT_D_puan.Value);
                    }

                }
            }
            int eslesmeyen_D_sayisi = Dler.Count - D_GT_eslesmeleri.Count;
            Toplam_puan += eslesmeyen_D_sayisi * (-1);
            return Toplam_puan;
        }
        public static double IntOUni(BoundingBox GT, BoundingBox D)
        {
            double IoU = 0;

            int x_left = Math.Max(GT.x0, D.x0);
            int y_bottom = Math.Max(GT.y0, D.y0);
            int x_right = Math.Min(GT.x1, D.x1);
            int y_top = Math.Min(GT.y1, D.y1);

            if ((x_right > x_left) && (y_top > y_bottom))
            {
                double Area_int = (x_right - x_left) * (y_top - y_bottom);
                IoU = Area_int / (GT.Area() + D.Area() + Area_int);
            }
            return IoU;
        }
        static List<KeyValuePair<GT_D_Cifti, double>> Puanlar_matrisi_olustur(List<BoundingBox> GTler, List<BoundingBox> Dler)
        {
            List<KeyValuePair<GT_D_Cifti, double>> puanlar = new List<KeyValuePair<GT_D_Cifti, double>>();
            for (int i_gt = 0; i_gt < GTler.Count; i_gt++)
            {
                for (int i_d = 0; i_d < Dler.Count; i_d++)
                {
                    if(GTler[i_gt].type == Dler[i_d].type)
                    {
                        double IoU = IntOUni(GTler[i_gt], Dler[i_d]);
                        puanlar.Add(new KeyValuePair<GT_D_Cifti, double>(new GT_D_Cifti(i_gt, i_d), IoU));
                    }
                    
                }
            }

            puanlar.Sort(delegate (KeyValuePair<GT_D_Cifti, double> v1, KeyValuePair<GT_D_Cifti, double> v2)
            {
                return v2.Value.CompareTo(v1.Value);
            });
            return puanlar;
        }
        static double Puan_hesapla(double IoU)
        {

            if (IoU >= 0.6)
            {
                return IoU * 3;
            }
            else
            {
                return (1 - IoU) * (-1);
            }
        }

    }
}
