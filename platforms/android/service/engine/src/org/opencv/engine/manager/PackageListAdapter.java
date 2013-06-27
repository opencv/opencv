package org.opencv.engine.manager;

import java.util.List;
import java.util.Map;

import android.content.Context;
import android.view.View;
import android.view.ViewGroup;
import android.widget.SimpleAdapter;

public class PackageListAdapter extends SimpleAdapter {

    public PackageListAdapter(Context context,
            List<? extends Map<String, ?>> data, int resource, String[] from,
            int[] to) {
        super(context, data, resource, from, to);
        // TODO Auto-generated constructor stub
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
      View view = super.getView(position, convertView, parent);
      @SuppressWarnings("unchecked")
      Map<String, String> item = (Map<String, String>)getItem(position);
      if (item.get("Activity") == "y")
      {
          view.setBackgroundColor(0x50ffffff);
      }
      else
      {
          view.setBackgroundColor(0xff000000);
      }

      return view;
    }
}
